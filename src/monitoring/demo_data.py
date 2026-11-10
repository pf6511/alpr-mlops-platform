"""
ALPR - Données de démonstration pour le monitoring.

    python -m src.monitoring.demo_data             # insère des passages fictifs
    python -m src.monitoring.demo_data --delete    # supprime UNIQUEMENT ces passages

Insère dans le journal des passages (table logs) des passages FICTIFS, marqués
source = 'demo' et plaques DEMO0001... :

    référence (jours -37 à -7) : situation normale
        bonne luminosité, confiance ≈ 0,9, ≈ 400 ms, peu d'incohérences,
        le modèle a presque toujours raison lors des validations
    récente (7 derniers jours) : dégradation
        images sombres (caméra de nuit, salissure), confiance ≈ 0,6, plus
        d'incohérences, le modèle se trompe souvent lors des validations

But : faire réagir le monitoring (dérive Evidently, seuils dépassés, alerte du DAG).
Les incohérences fictives sont créées DÉJÀ validées : elles n'apparaissent pas dans
la liste « en attente » du back office. Aucune image n'est créée (le réentraînement,
qui compte les images validées dans MinIO, n'est pas concerné).
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.settings import get_settings  # noqa: E402

DEMO_SOURCE = "demo"
BRANDS = ["renault", "peugeot", "citroen", "volkswagen", "toyota", "audi", "bmw", "dacia"]


def demo_passage(index: int, timestamp: datetime, degraded: bool, rng: random.Random) -> dict:
    """Un passage fictif ; `degraded` simule la période où la qualité se dégrade."""
    brand = rng.choice(BRANDS[:4] if not degraded else BRANDS)     # nouvelles marques en période récente
    mismatch = rng.random() < (0.30 if degraded else 0.05)
    authorized = rng.random() < 0.8
    passage = {
        "plaque": f"DEMO{index:04d}",
        "normalized_plate": f"DEMO{index:04d}",
        "timestamp": timestamp,
        "resultat": "autorisé" if authorized else "refusé",
        "motif": "AUTHORIZED" if authorized else "UNKNOWN_PLATE",
        "marque_predite": brand,
        "marque_confiance": round(rng.uniform(0.40, 0.75) if degraded else rng.uniform(0.80, 0.98), 3),
        "mismatch": mismatch,
        "ocr_confiance": round(rng.uniform(0.60, 0.85) if degraded else rng.uniform(0.85, 0.99), 3),
        "luminosite": round(rng.uniform(25, 70) if degraded else rng.uniform(100, 160), 1),
        "nettete": round(rng.uniform(40, 150) if degraded else rng.uniform(200, 500), 1),
        "duree_ms": int(rng.uniform(600, 1500) if degraded else rng.uniform(300, 600)),
        "source": DEMO_SOURCE,
        "modele_marque": "registry:vehicle_brand_model@champion (v1)",
        "validation_status": None,
        "validated_label": None,
        "validated_at": None,
    }
    if mismatch:   # incohérence déjà validée : le modèle se trompe souvent en période dégradée
        model_wrong = rng.random() < (0.60 if degraded else 0.05)
        passage["validation_status"] = "validated"
        passage["validated_label"] = rng.choice([b for b in BRANDS if b != brand]) if model_wrong else brand
        passage["validated_at"] = timestamp + timedelta(hours=1)
    return passage


def generate(now: datetime, n_reference: int = 150, n_current: int = 60, seed: int = 42) -> list:
    """Passages de la période de référence puis de la période récente (reproductibles : graine fixe)."""
    settings = get_settings().monitoring
    rng = random.Random(seed)
    current_start = now - timedelta(days=settings.current_days)
    reference_start = current_start - timedelta(days=settings.reference_days)

    def random_time(start: datetime, end: datetime) -> datetime:
        return start + (end - start) * rng.random()

    rows = [demo_passage(i, random_time(reference_start, current_start), False, rng) for i in range(n_reference)]
    rows += [demo_passage(n_reference + i, random_time(current_start, now), True, rng) for i in range(n_current)]
    return rows


def insert(rows: list, db) -> int:
    ph = db._placeholder()
    columns = list(rows[0])
    sql = f"INSERT INTO logs ({', '.join(columns)}) VALUES ({', '.join([ph] * len(columns))})"
    with db.get_cursor(commit=True) as cursor:
        for row in rows:
            cursor.execute(sql, tuple(row.values()))
    return len(rows)


def delete(db) -> int:
    ph = db._placeholder()
    with db.get_cursor(commit=True) as cursor:
        cursor.execute(f"DELETE FROM logs WHERE source = {ph}", (DEMO_SOURCE,))
        return cursor.rowcount


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Passages fictifs pour démontrer le monitoring")
    parser.add_argument("--delete", action="store_true", help="Supprimer les passages de démonstration")
    args = parser.parse_args(argv)

    from src.data.database import DatabaseManager

    db = DatabaseManager()
    if args.delete:
        print(f"🧹 {delete(db)} passage(s) de démonstration supprimé(s)")
        return 0
    delete(db)                                  # pas de doublons si on relance
    count = insert(generate(datetime.now()), db)
    print(f"✅ {count} passages de démonstration insérés (source = '{DEMO_SOURCE}')")
    print("   Rapport : python -m src.monitoring.report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
