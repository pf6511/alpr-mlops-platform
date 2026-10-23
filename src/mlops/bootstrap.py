"""
Initialisation du registre : le modèle local actuel devient la version « champion ».

    python -m src.mlops.bootstrap            # n'agit que si aucun champion n'existe
    python -m src.mlops.bootstrap --force    # enregistre une nouvelle version malgré tout

Le manifeste consigne honnêtement l'origine : le jeu d'entraînement de ce
modèle est inconnu (aucune trace dans le code, la documentation ou l'historique).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import Settings, get_settings  # noqa: E402
from src.inference.loader import read_local_checkpoint, resolve_model_path  # noqa: E402
from src.mlops.bundle import write_bundle  # noqa: E402
from src.mlops.registry import BrandModelRegistry, RegisteredBrandModel  # noqa: E402

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(settings: Optional[Settings] = None, registry: Optional[BrandModelRegistry] = None,
              force: bool = False) -> Optional[RegisteredBrandModel]:
    settings = settings or get_settings()
    registry = registry or BrandModelRegistry(settings.mlflow)
    alias = settings.mlflow.champion_alias

    existing = registry.get(alias)
    if existing and not force:
        logger.info("Un champion existe déjà (version %s) : rien à faire.", existing.version)
        return existing

    path = resolve_model_path([settings.models.efficientnet_path])
    if path is None:
        raise FileNotFoundError(f"Modèle local introuvable : {settings.models.efficientnet_path}")

    state_dict, classes, info = read_local_checkpoint(path, settings.models.brand_classes)
    manifest = {
        "origin": "bootstrap",
        "source_file": path.name,
        "source_sha256": sha256(path),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "status": "unknown",
            "note": "Jeu d'entraînement d'origine introuvable : aucune trace dans le code, "
                    "la documentation ou l'historique Git. Lignage à reconstituer.",
        },
        "checkpoint_info": info,
    }
    metrics = {}
    if info.get("val_acc") is not None:
        val_acc = float(info["val_acc"])
        metrics["reported_val_accuracy"] = val_acc / 100 if val_acc > 1 else val_acc

    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = write_bundle(Path(tmp) / "bundle", state_dict, classes, manifest)
        return registry.register(
            bundle_dir,
            run_name="bootstrap-modele-local",
            params={"origin": "bootstrap", "source_file": path.name, "num_classes": len(classes),
                    "epoch": info.get("epoch", "inconnue")},
            metrics=metrics,
            tags={"origin": "bootstrap", "dataset_status": "unknown"},
            alias=alias,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Enregistre le modèle local comme champion du registre")
    parser.add_argument("--force", action="store_true", help="Enregistrer même si un champion existe")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model = bootstrap(force=args.force)
    print(f"✅ {get_settings().mlflow.brand_model_name} : version {model.version} = champion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
