"""
Cycle MLOps complet avec de vrais composants (exécuté en CI, où torch et MLflow
sont installés) : MLflow local (SQLite), EfficientNet B0, stockage local.

    modèle local → bootstrap (v1 champion)
      → images validées → entraînement (v2 challenger)
      → promotion (même jeu de test) → le moteur charge le champion du registre
"""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
mlflow = pytest.importorskip("mlflow")

from PIL import Image  # noqa: E402

from configs.settings import get_settings  # noqa: E402
from src.data.storage import StorageManager  # noqa: E402
from src.inference.loader import build_brand_network, load_brand_bundle  # noqa: E402
from src.mlops.bootstrap import bootstrap  # noqa: E402
from src.mlops.promote import run_promotion  # noqa: E402
from src.mlops.registry import BrandModelRegistry  # noqa: E402
from src.training.train import run_training  # noqa: E402


@pytest.fixture
def mlops_env(tmp_path):
    settings = copy.deepcopy(get_settings())
    classes = list(settings.models.brand_classes)

    # Modèle local factice au format historique (poids aléatoires, 22 classes)
    local_model = tmp_path / "best_model_efficientNet_finetune.pth"
    torch.save({"model_state_dict": build_brand_network(len(classes)).state_dict(),
                "label_encoder_classes": classes, "epoch": 18, "val_acc": 82.6}, local_model)
    settings.models.efficientnet_path = str(local_model)

    # MLflow local : base SQLite et artefacts dans le dossier temporaire
    settings.mlflow.tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings.mlflow.experiment = "test-brand"
    settings.mlflow.cache_dir = str(tmp_path / "cache")
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.create_experiment("test-brand", artifact_location=(tmp_path / "artifacts").as_uri())

    # Stockage local avec des images « validées » de deux marques
    storage = StorageManager()
    storage.mode, storage.s3_client = "local", None
    storage.local_paths = {n: tmp_path / n for n in ("dataset", "captures", "artifacts")}
    rng = np.random.default_rng(0)
    for brand, color in (("renault", (220, 200, 0)), ("peugeot", (0, 60, 200))):
        folder = storage.local_paths["dataset"] / "labeled" / brand
        folder.mkdir(parents=True)
        for i in range(12):
            pixels = np.clip(np.array(color) + rng.integers(-30, 30, (64, 96, 3)), 0, 255).astype("uint8")
            Image.fromarray(pixels).save(folder / f"{i:02d}.jpg")

    registry = BrandModelRegistry(settings.mlflow)
    return settings, storage, registry


def test_full_mlops_cycle(mlops_env):
    settings, storage, registry = mlops_env

    # 1. Bootstrap : le modèle local devient la version 1, champion
    v1 = bootstrap(settings, registry)
    assert v1.version == "1" and registry.get("champion").version == "1"
    assert bootstrap(settings, registry).version == "1"            # idempotent

    # 2. Entraînement : nouvelle version challenger, avec manifeste de lignage
    summary = run_training(settings, storage, registry, test_fraction=0.5, epochs=1,
                           batch_size=4, min_train_images=1)
    assert summary["version"] == "2" and registry.get("challenger").version == "2"
    assert summary["counts"]["train"] + summary["counts"]["test"] == 24

    bundle = registry.download("challenger")
    manifest = json.loads((bundle.directory / "manifest.json").read_text())
    assert manifest["architecture"] == "efficientnet_b0"
    assert manifest["parent"].endswith("(v1)")
    assert len(manifest["dataset"]["test_set"]) == summary["counts"]["test"] > 0
    assert set(manifest["evaluation"]) == {"challenger", "champion"}

    # 3. Promotion : décision cohérente avec la règle, alias déplacé si promu
    report = run_promotion(settings, storage, registry, min_test_images=1)
    assert report["test_images"] == summary["counts"]["test"]
    expected_champion = "2" if report["promoted"] else "1"
    assert registry.get("champion").version == expected_champion

    # 4. Le moteur charge le champion du registre (poids sans code, weights_only)
    classifier = load_brand_bundle(registry.download().directory)
    assert classifier is not None and len(classifier.classes) == len(settings.models.brand_classes)
