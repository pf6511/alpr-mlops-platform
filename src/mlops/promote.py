"""
Promotion : le challenger devient-il champion ?

    python -m src.mlops.promote
    python -m src.mlops.promote --min-test-images 5     # démonstration sur peu de données

Les deux modèles sont réévalués maintenant, sur le MÊME jeu de test : celui
inscrit dans le manifeste du challenger. La décision suit une règle explicite
(decide_promotion), puis l'alias « champion » est déplacé ou non.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import Settings, get_settings  # noqa: E402
from src.inference.loader import read_bundle  # noqa: E402
from src.mlops.registry import PROMOTION_TAG, BrandModelRegistry  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class PromotionDecision:
    promote: bool
    reason: str


def decide_promotion(champion: Dict[str, float], challenger: Dict[str, float],
                     min_test_images: int = 20, f1_tolerance: float = 0.0) -> PromotionDecision:
    """
    Règle de promotion :
      - un jeu de test d'au moins `min_test_images` images ;
      - une précision au moins égale à celle du champion ;
      - un F1 macro qui ne recule pas de plus de `f1_tolerance`.
    """
    n = challenger.get("n", 0)
    if n < min_test_images:
        return PromotionDecision(False, f"jeu de test trop petit ({n} < {min_test_images} images)")
    if challenger["accuracy"] < champion["accuracy"]:
        return PromotionDecision(False, f"précision en baisse ({challenger['accuracy']:.3f} < {champion['accuracy']:.3f})")
    if challenger["macro_f1"] < champion["macro_f1"] - f1_tolerance:
        return PromotionDecision(False, f"F1 macro en baisse ({challenger['macro_f1']:.3f} < {champion['macro_f1']:.3f})")
    return PromotionDecision(True, f"précision {challenger['accuracy']:.3f} ≥ {champion['accuracy']:.3f} "
                                   f"sur {n} images de test")


def _test_items(manifest: Dict[str, Any], classes, storage, work_dir: Path):
    """Reconstruit le jeu de test du manifeste ; ignore les images devenues introuvables."""
    items, missing = [], 0
    for i, entry in enumerate(manifest.get("dataset", {}).get("test_set", [])):
        if entry["label"] not in classes:
            missing += 1
            continue
        if entry["origin"] == "validated":
            dest = work_dir / f"{i:06d}{Path(entry['id']).suffix}"
            ok, _ = storage.download_file(entry["id"], dest, bucket_type="dataset")
            path = dest if ok else None
        else:
            path = Path(entry["id"]) if Path(entry["id"]).exists() else None
        if path is None:
            missing += 1
            continue
        items.append((str(path), classes.index(entry["label"])))
    return items, missing


def run_promotion(settings: Optional[Settings] = None, storage=None, registry: Optional[BrandModelRegistry] = None,
                  min_test_images: int = 20, f1_tolerance: float = 0.0) -> Dict[str, Any]:
    from src.training import trainer
    from src.training.metrics import classification_metrics

    settings = settings or get_settings()
    cfg = settings.mlflow
    registry = registry or BrandModelRegistry(cfg)
    if storage is None:
        from src.data.storage import get_storage
        storage = get_storage()

    champion_ref, challenger_ref = registry.get(cfg.champion_alias), registry.get(cfg.challenger_alias)
    if challenger_ref is None:
        return {"promoted": False, "reason": "aucun challenger dans le registre"}
    if champion_ref is not None and champion_ref.version == challenger_ref.version:
        return {"promoted": False, "reason": f"le challenger (v{challenger_ref.version}) est déjà champion"}

    challenger_state, classes, manifest = read_bundle(registry.download(cfg.challenger_alias).directory)
    champion_state, champion_classes, _ = read_bundle(registry.download(cfg.champion_alias).directory)
    if list(champion_classes) != list(classes):
        reason = "classes différentes entre champion et challenger"
        registry.tag_version(challenger_ref.version, PROMOTION_TAG, f"rejected: {reason}")
        return {"promoted": False, "reason": reason}

    with tempfile.TemporaryDirectory() as tmp:
        items, missing = _test_items(manifest, classes, storage, Path(tmp))
        targets = [t for _, t in items]
        challenger_m = classification_metrics(trainer.predict(challenger_state, classes, items), targets, classes)
        champion_m = classification_metrics(trainer.predict(champion_state, classes, items), targets, classes)

    decision = decide_promotion(champion_m, challenger_m, min_test_images, f1_tolerance)
    status = "promoted" if decision.promote else "rejected"
    registry.tag_version(challenger_ref.version, PROMOTION_TAG, f"{status}: {decision.reason}")
    if decision.promote:
        registry.set_alias(cfg.champion_alias, challenger_ref.version)
        if champion_ref is not None:
            registry.tag_version(champion_ref.version, "replaced_by", challenger_ref.version)

    return {
        "promoted": decision.promote, "reason": decision.reason,
        "challenger_version": challenger_ref.version,
        "champion_version_before": champion_ref.version if champion_ref else None,
        "test_images": len(items), "test_images_missing": missing,
        "metrics": {"challenger": challenger_m, "champion": champion_m},
        "decision": asdict(decision),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Promotion challenger → champion")
    parser.add_argument("--min-test-images", type=int, default=20)
    parser.add_argument("--f1-tolerance", type=float, default=0.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = run_promotion(min_test_images=args.min_test_images, f1_tolerance=args.f1_tolerance)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(("🏆 Promu : " if report["promoted"] else "⏸️ Non promu : ") + report["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
