"""
ALPR - Rapport de monitoring de la solution en production.

    python -m src.monitoring.report                 # période récente vs référence
    python -m src.monitoring.report --no-drift      # indicateurs seuls (sans Evidently)

Les briques, dans l'ordre :

    1. COLLECTE      chaque passage est journalisé avec ses mesures (table logs)
    2. LECTURE       la vue monitoring.passages (le journal SANS les plaques)
    3. PÉRIODES      récente (current_days) / référence (les reference_days d'avant)
    4. INDICATEURS   service, données, modèle (calculs simples, pandas)
    5. SEUILS        critères d'acceptation du cahier des charges (config.yaml)
    6. DÉRIVE        rapport Evidently (HTML) : les données récentes ressemblent-elles
                     à celles de la référence ?
    7. RESTITUTION   rapport HTML + une ligne d'historique (data/monitoring/history.jsonl)

Le taux d'erreur du modèle de marque vient des validations de l'opérateur : pour une
incohérence validée, le modèle avait raison si la marque retenue est la marque
prédite. Limite : seules les incohérences sont validées (indicateur partiel).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.settings import MonitoringSettings, get_settings  # noqa: E402
from src.inference.brands import normalize_brand  # noqa: E402

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data") / "monitoring"
HISTORY_FILE = REPORTS_DIR / "history.jsonl"
RESULT_PREFIX = "ALPR_RESULT "          # dernière ligne de sortie, lue par le DAG Airflow

# Colonnes comparées par le rapport de dérive (aucune donnée personnelle)
NUMERICAL_COLUMNS = ["marque_confiance", "ocr_confiance", "luminosite", "nettete", "duree_ms"]
CATEGORICAL_COLUMNS = ["marque_predite", "motif", "source"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LECTURE
# ═══════════════════════════════════════════════════════════════════════════════

def load_passages(since: datetime, db=None) -> pd.DataFrame:
    """Passages depuis `since`, lus dans la vue monitoring.passages (sans plaques)."""
    if db is None:
        from src.data.database import DatabaseManager

        db = DatabaseManager()
    with db.get_cursor() as cursor:
        cursor.execute("SELECT * FROM monitoring.passages WHERE timestamp >= %s", (since,))
        rows = cursor.fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PÉRIODES
# ═══════════════════════════════════════════════════════════════════════════════

def split_periods(df: pd.DataFrame, now: datetime,
                  settings: MonitoringSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(référence, récente) : [now - current - reference, now - current[ et [now - current, now]."""
    if df.empty:
        return df, df
    current_start = now - timedelta(days=settings.current_days)
    reference_start = current_start - timedelta(days=settings.reference_days)
    reference = df[(df["timestamp"] >= reference_start) & (df["timestamp"] < current_start)]
    current = df[df["timestamp"] >= current_start]
    return reference, current


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INDICATEURS
# ═══════════════════════════════════════════════════════════════════════════════

def _mean(series: pd.Series) -> Optional[float]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(series.mean()), 4) if len(series) else None


def brand_error_rate(df: pd.DataFrame) -> Tuple[Optional[float], int]:
    """(taux d'erreur, nombre de validations) : le modèle se trompe si la marque retenue ≠ prédite."""
    if df.empty:
        return None, 0
    validated = df[df["validation_status"] == "validated"].dropna(subset=["validated_label", "marque_predite"])
    if validated.empty:
        return None, 0
    wrong = validated.apply(
        lambda r: normalize_brand(r["validated_label"]) != normalize_brand(r["marque_predite"]), axis=1)
    return round(float(wrong.mean()), 4), len(validated)


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Indicateurs d'une période : service, données, modèle."""
    if df.empty:
        return {"passages": 0}
    durations = pd.to_numeric(df["duree_ms"], errors="coerce").dropna()
    error_rate, validations = brand_error_rate(df)
    return {
        # Service
        "passages": len(df),
        "part_refus": round(float((df["resultat"] == "refusé").mean()), 4),
        "duree_ms_p95": int(durations.quantile(0.95)) if len(durations) else None,
        # Données
        "luminosite_moyenne": _mean(df["luminosite"]),
        "nettete_moyenne": _mean(df["nettete"]),
        # Modèle
        "confiance_marque_moyenne": _mean(df["marque_confiance"]),
        "confiance_ocr_moyenne": _mean(df["ocr_confiance"]),
        "part_incoherences": round(float(df["mismatch"].fillna(False).astype(bool).mean()), 4),
        "validations": validations,
        "taux_erreur_marque": error_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SEUILS (critères d'acceptation du cahier des charges)
# ═══════════════════════════════════════════════════════════════════════════════

def check_thresholds(indicators: Dict[str, Any], settings: MonitoringSettings) -> List[str]:
    """Liste des alertes (vide si tous les critères sont respectés ou non mesurables)."""
    alerts = []
    rate = indicators.get("taux_erreur_marque")
    if rate is not None and rate > settings.max_brand_error_rate:
        alerts.append(f"taux d'erreur du modèle de marque {rate:.0%} > {settings.max_brand_error_rate:.0%}")
    confidence = indicators.get("confiance_marque_moyenne")
    if confidence is not None and confidence < settings.min_brand_confidence:
        alerts.append(f"confiance moyenne du modèle de marque {confidence:.2f} < {settings.min_brand_confidence:.2f}")
    p95 = indicators.get("duree_ms_p95")
    if p95 is not None and p95 > settings.max_duration_ms_p95:
        alerts.append(f"temps de traitement (p95) {p95} ms > {settings.max_duration_ms_p95} ms")
    return alerts


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DÉRIVE (Evidently)
# ═══════════════════════════════════════════════════════════════════════════════

def drift_report(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path) -> Optional[float]:
    """
    Rapport Evidently « DataDriftPreset » : compare colonne par colonne la période
    récente à la référence (test statistique choisi automatiquement par Evidently).
    Enregistre le HTML et renvoie la part de colonnes en dérive (ou None si illisible).
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    columns = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    definition = DataDefinition(numerical_columns=NUMERICAL_COLUMNS, categorical_columns=CATEGORICAL_COLUMNS)
    current_ds = Dataset.from_pandas(current[columns].copy(), data_definition=definition)
    reference_ds = Dataset.from_pandas(reference[columns].copy(), data_definition=definition)

    snapshot = Report([DataDriftPreset()]).run(current_ds, reference_ds)   # (récente, référence)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    return drifted_share(snapshot.dict())


def drifted_share(result: Dict[str, Any]) -> Optional[float]:
    """
    Part de colonnes en dérive, lue dans le résultat Evidently : la métrique
    DriftedColumnsCount, dont la valeur est {"count": ..., "share": ...}. Le nom de la
    métrique est cherché dans toute sa description (son emplacement varie selon les versions).
    """
    for metric in result.get("metrics", []):
        value = metric.get("value")
        description = json.dumps({k: v for k, v in metric.items() if k != "value"}, default=str)
        if "DriftedColumnsCount" in description and isinstance(value, dict) and "share" in value:
            return round(float(value["share"]), 4)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run(now: Optional[datetime] = None, with_drift: bool = True, db=None,
        settings: Optional[MonitoringSettings] = None) -> Dict[str, Any]:
    """Un rapport complet : lecture → périodes → indicateurs → seuils → dérive → historique."""
    settings = settings or get_settings().monitoring
    now = now or datetime.now()
    since = now - timedelta(days=settings.current_days + settings.reference_days)

    reference, current = split_periods(load_passages(since, db=db), now, settings)
    current_ind, reference_ind = compute_indicators(current), compute_indicators(reference)
    alerts = check_thresholds(current_ind, settings)

    report: Dict[str, Any] = {
        "date": now.isoformat(timespec="seconds"),
        "periode_recente_jours": settings.current_days,
        "periode_reference_jours": settings.reference_days,
        "recente": current_ind,
        "reference": reference_ind,
        "alertes": alerts,
        "derive": None,
        "rapport_html": None,
    }

    enough = min(len(current), len(reference)) >= settings.min_passages
    if with_drift and enough:
        html_path = REPORTS_DIR / f"derive_{now:%Y%m%d_%H%M}.html"
        try:
            share = drift_report(reference, current, html_path)
            report["derive"] = {"part_colonnes_en_derive": share}
            report["rapport_html"] = str(html_path)
            if share is not None and share >= settings.max_drift_share:
                alerts.append(f"dérive des données : {share:.0%} des colonnes ≥ {settings.max_drift_share:.0%}"
                              f" (voir le rapport Evidently)")
        except Exception as exc:  # noqa: BLE001 - le rapport de dérive ne doit pas bloquer les indicateurs
            logger.error("Rapport de dérive impossible : %s", exc)
            report["derive"] = {"erreur": str(exc)}
    elif with_drift:
        report["derive"] = {"ignore": f"moins de {settings.min_passages} passages dans une des périodes "
                                      f"(récente : {len(current)}, référence : {len(reference)})"}

    report["statut"] = "alerte" if alerts else "ok"
    _append_history(report)
    return report


def _append_history(report: Dict[str, Any]) -> None:
    """Une ligne par exécution : l'évolution des indicateurs dans le temps."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rapport de monitoring de la solution ALPR")
    parser.add_argument("--no-drift", action="store_true", help="Indicateurs seuls, sans rapport Evidently")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    report = run(with_drift=not args.no_drift)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(RESULT_PREFIX + json.dumps(report, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
