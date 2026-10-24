"""
Réentraînement du classificateur de marque → nouvelle version « challenger ».

    python -m src.training.train --epochs 3
    python -m src.training.train --base-dir /data/vmmrdb_par_marque --max-per-class 200

Étapes :
  1. part des poids du « champion » actuel (affinage, mêmes classes)
  2. rassemble les images validées en production (+ un socle facultatif)
  3. répartit entraînement / test de façon stable, rédige le manifeste de lignage
  4. entraîne, puis évalue le challenger ET le champion sur le même jeu de test
  5. enregistre la version dans MLflow avec l'alias « challenger »

La promotion en champion est une décision séparée : python -m src.mlops.promote
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import Settings, get_settings  # noqa: E402
from src.inference.loader import read_bundle  # noqa: E402
from src.mlops.bundle import write_bundle  # noqa: E402
from src.mlops.registry import VALIDATED_IMAGES_TAG, BrandModelRegistry  # noqa: E402
from src.training import trainer  # noqa: E402
from src.training.dataset import build_manifest, collect_base, collect_validated, materialize, split_samples  # noqa: E402
from src.training.metrics import classification_metrics  # noqa: E402

logger = logging.getLogger(__name__)


class NotEnoughData(RuntimeError):
    """Pas assez d'images pour entraîner : aucune version n'est créée."""


def _items(samples, classes, split):
    return [(str(s.local_path), classes.index(s.label)) for s in samples if s.split == split]


def run_training(settings: Optional[Settings] = None, storage=None, registry: Optional[BrandModelRegistry] = None,
                 base_dir: Optional[Path] = None, max_per_class: Optional[int] = None,
                 test_fraction: float = 0.2, epochs: int = 3, lr: float = 1e-4, batch_size: int = 16,
                 freeze_backbone: bool = True, min_train_images: int = 10) -> Dict[str, Any]:
    settings = settings or get_settings()
    registry = registry or BrandModelRegistry(settings.mlflow)
    if storage is None:
        from src.data.storage import get_storage
        storage = get_storage()

    # 1. Point de départ : le champion
    champion = registry.download(settings.mlflow.champion_alias)
    if champion is None:
        raise RuntimeError("Aucun champion dans le registre : lancer d'abord « python -m src.mlops.bootstrap »")
    champion_state, classes, _ = read_bundle(champion.directory)

    # 2-3. Données, répartition, manifeste
    samples = collect_validated(storage) + collect_base(base_dir, max_per_class)
    kept, rejected = split_samples(samples, classes, test_fraction)

    with tempfile.TemporaryDirectory() as tmp:
        ready = materialize(kept, storage, Path(tmp) / "data")
        train_items, test_items = _items(ready, classes, "train"), _items(ready, classes, "test")
        if len(train_items) < min_train_images:
            raise NotEnoughData(f"{len(train_items)} image(s) d'entraînement (minimum {min_train_images})")

        manifest = build_manifest(
            ready, rejected, classes, test_fraction=test_fraction,
            sources={"validated": f"{settings.s3.bucket_dataset}/labeled/",
                     "base": str(base_dir) if base_dir else "aucun"},
        )

        # 4. Entraînement, puis évaluation des deux modèles sur le même jeu de test
        params = {"epochs": epochs, "lr": lr, "batch_size": batch_size, "freeze_backbone": freeze_backbone,
                  "test_fraction": test_fraction, "max_per_class": max_per_class or "toutes",
                  "parent": champion.label, "dataset_id": manifest["dataset_id"]}
        new_state, history = trainer.train(champion_state, classes, train_items, epochs=epochs, lr=lr,
                                           batch_size=batch_size, freeze_backbone=freeze_backbone)
        targets = [t for _, t in test_items]
        challenger_m = classification_metrics(trainer.predict(new_state, classes, test_items), targets, classes)
        champion_m = classification_metrics(trainer.predict(champion_state, classes, test_items), targets, classes)

        # 5. Enregistrement : paquet + lignage
        bundle = write_bundle(Path(tmp) / "bundle", new_state, classes, {
            "origin": "training", "parent": champion.label, "training": params, "dataset": manifest,
            "evaluation": {"challenger": challenger_m, "champion": champion_m},
        })
        metrics = {
            "test_accuracy": challenger_m["accuracy"], "test_top3_accuracy": challenger_m["top3_accuracy"],
            "test_macro_f1": challenger_m["macro_f1"], "champion_test_accuracy": champion_m["accuracy"],
            "champion_test_macro_f1": champion_m["macro_f1"],
            "train_images": len(train_items), "test_images": len(test_items),
        }
        version = registry.register(
            bundle, run_name=f"train-{manifest['dataset_id']}", params=params, metrics=metrics, history=history,
            tags={"origin": "training", "dataset_id": manifest["dataset_id"], "parent": champion.label,
                  # mémoire du réentraînement automatique : images validées vues par cette version
                  VALIDATED_IMAGES_TAG: str(manifest["counts"]["by_origin"].get("validated", 0))},
            alias=settings.mlflow.challenger_alias,
        )

    return {"version": version.version, "run_id": version.run_id, "dataset_id": manifest["dataset_id"],
            "metrics": metrics, "counts": manifest["counts"], "rejected": manifest["rejected_unknown_labels"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Réentraîne le classificateur de marque (→ challenger)")
    parser.add_argument("--base-dir", type=Path, help="Socle local <dossier>/<marque>/*.jpg")
    parser.add_argument("--max-per-class", type=int, help="Images du socle par marque (échantillon)")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--full-finetune", action="store_true", help="Entraîner tout le réseau (GPU conseillé)")
    parser.add_argument("--min-train-images", type=int, default=10)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        summary = run_training(base_dir=args.base_dir, max_per_class=args.max_per_class,
                               test_fraction=args.test_fraction, epochs=args.epochs, lr=args.lr,
                               batch_size=args.batch_size, freeze_backbone=not args.full_finetune,
                               min_train_images=args.min_train_images)
    except NotEnoughData as exc:
        print(f"⚠️ Entraînement non lancé : {exc}")
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
