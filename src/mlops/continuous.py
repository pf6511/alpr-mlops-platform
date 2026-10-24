"""
Réentraînement automatique du classificateur de marque.

    python -m src.mlops.continuous            # un cycle (déclenché par un ordonnanceur)
    python -m src.mlops.continuous --loop     # cycles périodiques (service Docker alpr-retrain)
    python -m src.mlops.continuous --force    # un cycle, même sans nouvelles images
    python -m src.mlops.continuous --check    # décision seule : faut-il un cycle ? (DAG Airflow)

Un cycle :
  0. un challenger attend encore sa promotion (cycle précédent interrompu) ? → promotion
  1. compte les images validées par les opérateurs (bucket dataset, labeled/<marque>/)
  2. compare au nombre utilisé par le dernier entraînement : c'est le REGISTRE qui
     s'en souvient (tag « validated_images » posé sur chaque version entraînée)
  3. pas assez de nouvelles images → rien à faire (cycle « skipped »)
  4. sinon : entraînement (→ challenger), puis promotion (même jeu de test, règle explicite)
  5. rapport journalisé (data/retrain_history.jsonl), et la décision est tracée dans MLflow

Un cycle est idempotent : relancé sans nouvelles images, il ne fait rien. Un verrou
empêche deux cycles simultanés (ordonnanceur + lancement manuel). L'application
charge le nouveau champion d'elle-même (src/services/system.py, ChampionWatcher).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import Settings, get_settings  # noqa: E402
from src.mlops.registry import PROMOTION_TAG, VALIDATED_IMAGES_TAG, BrandModelRegistry  # noqa: E402
from src.training.dataset import collect_validated  # noqa: E402

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
LOCK_FILE = DATA_DIR / "retrain.lock"
# Dernière ligne de la sortie : le résultat en JSON compact, lu par le DAG Airflow
RESULT_PREFIX = "ALPR_RESULT "
HISTORY_FILE = DATA_DIR / "retrain_history.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# DÉCLENCHEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TriggerDecision:
    run: bool
    reason: str
    validated_images: int
    last_trained: int

    @property
    def new_images(self) -> int:
        return self.validated_images - self.last_trained


def count_validated(storage, classes) -> int:
    """Images validées dont la marque fait partie des classes du modèle."""
    known = set(classes)
    return sum(1 for s in collect_validated(storage) if s.label in known)


def last_trained_count(registry: BrandModelRegistry) -> int:
    """Images validées utilisées par l'entraînement le plus récent (0 si aucun)."""
    counts = [int(v.tags[VALIDATED_IMAGES_TAG]) for v in registry.versions()
              if str(v.tags.get(VALIDATED_IMAGES_TAG, "")).isdigit()]
    return max(counts, default=0)


def decide_trigger(validated_images: int, last_trained: int, min_new_images: int,
                   force: bool = False) -> TriggerDecision:
    new = validated_images - last_trained
    if force:
        return TriggerDecision(True, "cycle forcé", validated_images, last_trained)
    if new >= min_new_images:
        return TriggerDecision(True, f"{new} nouvelle(s) image(s) validée(s) ≥ {min_new_images}",
                               validated_images, last_trained)
    return TriggerDecision(False, f"{new} nouvelle(s) image(s) validée(s) < {min_new_images} : rien à faire",
                           validated_images, last_trained)


def check(settings: Optional[Settings] = None, storage=None,
          registry: Optional[BrandModelRegistry] = None) -> Dict[str, Any]:
    """Faut-il lancer un cycle ? Même règle que run_cycle, sans rien entraîner (lecture seule)."""
    settings = settings or get_settings()
    registry = registry or BrandModelRegistry(settings.mlflow)
    if storage is None:
        from src.data.storage import get_storage
        storage = get_storage()
    if pending_challenger(registry, settings) is not None:
        return {"run": True, "reason": "challenger en attente de promotion"}
    decision = decide_trigger(count_validated(storage, settings.models.brand_classes),
                              last_trained_count(registry), settings.continuous.min_new_images)
    return asdict(decision) | {"new_images": decision.new_images}


def retraining_progress(settings: Settings, storage, registry: BrandModelRegistry) -> Dict[str, Any]:
    """Où en est le prochain réentraînement ? (affiché dans l'onglet Validation)."""
    decision = decide_trigger(count_validated(storage, settings.models.brand_classes),
                              last_trained_count(registry), settings.continuous.min_new_images)
    return {"new_images": decision.new_images, "threshold": settings.continuous.min_new_images,
            "validated_images": decision.validated_images, "will_retrain": decision.run}


# ═══════════════════════════════════════════════════════════════════════════════
# CYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def cycle_lock(path: Optional[Path] = None):
    """Verrou exclusif (libéré automatiquement si le processus s'arrête)."""
    path = Path(path or LOCK_FILE)
    try:
        import fcntl
    except ImportError:          # Windows hors conteneur : pas de verrou
        yield True
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_cycle(settings: Optional[Settings] = None, storage=None, registry: Optional[BrandModelRegistry] = None,
              force: bool = False, train_fn: Optional[Callable] = None,
              promote_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Exécute un cycle et renvoie son rapport. `train_fn` et `promote_fn` : injection (tests)."""
    settings = settings or get_settings()
    cfg = settings.continuous
    registry = registry or BrandModelRegistry(settings.mlflow)
    if storage is None:
        from src.data.storage import get_storage
        storage = get_storage()
    if train_fn is None:
        from src.training.train import run_training as train_fn
    if promote_fn is None:
        from src.mlops.promote import run_promotion as promote_fn
    from src.training.train import NotEnoughData

    report: Dict[str, Any] = {"started_at": datetime.now(timezone.utc).isoformat()}

    # 0. Un challenger n'a jamais été jugé (cycle précédent interrompu) : on le juge d'abord
    if pending_challenger(registry, settings) is not None:
        report["trigger"] = {"reason": "challenger en attente de promotion"}
        return _promote(report, settings, storage, registry, promote_fn)

    # 1-3. Faut-il réentraîner ?
    decision = decide_trigger(count_validated(storage, settings.models.brand_classes),
                              last_trained_count(registry), cfg.min_new_images, force)
    report["trigger"] = asdict(decision) | {"new_images": decision.new_images}
    if not decision.run:
        report["status"] = "skipped"
        return _finish(report)

    # 4. Entraînement → challenger
    try:
        report["training"] = train_fn(settings, storage, registry, epochs=cfg.epochs,
                                      min_train_images=cfg.min_train_images)
    except NotEnoughData as exc:
        report["status"], report["reason"] = "not_enough_data", str(exc)
        return _finish(report)

    return _promote(report, settings, storage, registry, promote_fn)


def pending_challenger(registry: BrandModelRegistry, settings: Settings):
    """Challenger sans décision de promotion, ou None."""
    challenger = registry.get(settings.mlflow.challenger_alias)
    if challenger is None or PROMOTION_TAG in challenger.tags:
        return None
    champion = registry.get(settings.mlflow.champion_alias)
    if champion is not None and champion.version == challenger.version:
        return None
    return challenger


def _promote(report, settings, storage, registry, promote_fn) -> Dict[str, Any]:
    """Promotion → champion ou rejet (la raison est tracée dans MLflow par run_promotion)."""
    promotion = promote_fn(settings, storage, registry, min_test_images=settings.continuous.min_test_images)
    report["promotion"] = {k: promotion.get(k) for k in
                           ("promoted", "reason", "challenger_version", "champion_version_before", "test_images")}
    report["status"] = "promoted" if promotion.get("promoted") else "not_promoted"
    return _finish(report)


def _finish(report: Dict[str, Any]) -> Dict[str, Any]:
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("Cycle de réentraînement : %s (%s)", report["status"],
                report.get("promotion", {}).get("reason") or report.get("reason")
                or report.get("trigger", {}).get("reason"))
    return report


def record(report: Dict[str, Any], history_file: Optional[Path] = None) -> None:
    """Ajoute le rapport à l'historique des cycles (une ligne JSON par cycle)."""
    history_file = Path(history_file or HISTORY_FILE)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")


def guarded_cycle(force: bool = False, **kwargs) -> Dict[str, Any]:
    """Un cycle sous verrou, avec journalisation ; les erreurs sont rapportées, pas propagées."""
    with cycle_lock() as acquired:
        if not acquired:
            report = {"status": "locked", "reason": "un cycle est déjà en cours"}
        else:
            try:
                report = run_cycle(force=force, **kwargs)
            except Exception as exc:  # noqa: BLE001 - registre ou stockage injoignable
                logger.exception("Cycle de réentraînement en échec")
                report = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    report.setdefault("finished_at", datetime.now(timezone.utc).isoformat())
    record(report)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def emit(result: Dict[str, Any]) -> None:
    """Affiche le résultat lisiblement, puis sur une ligne compacte pour les ordonnanceurs."""
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False, default=str), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Réentraînement automatique du classificateur de marque")
    parser.add_argument("--loop", action="store_true", help="Cycles périodiques (continuous_training.interval_hours)")
    parser.add_argument("--force", action="store_true", help="Réentraîner même sans nouvelles images")
    parser.add_argument("--check", action="store_true", help="Décision seule (aucun entraînement)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.check:
        emit(check())
        return 0
    if not args.loop:
        report = guarded_cycle(force=args.force)
        emit(report)
        return 1 if report["status"] == "error" else 0

    interval_s = get_settings().continuous.interval_hours * 3600
    logger.info("Réentraînement automatique : un cycle toutes les %.2f h", interval_s / 3600)
    while True:
        guarded_cycle()
        time.sleep(interval_s)


if __name__ == "__main__":
    sys.exit(main())
