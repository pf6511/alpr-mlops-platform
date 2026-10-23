"""
Chargement des modèles du cœur d'inférence.

Point unique où sont décidés les fichiers chargés, les replis éventuels et
l'architecture du classificateur de marque. Les trois points d'entrée
(Gradio, CLI, Ray Serve) obtiennent donc exactement les mêmes modèles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

from configs.settings import Settings, get_settings

from .models import (
    BrandClassifier,
    Detector,
    FastPlateOcrReader,
    PlateReader,
    SimpleYoloDetector,
    TorchBrandClassifier,
    UltralyticsDetector,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Architecture du classificateur de marque : la SEULE définition du projet.
# Le checkpoint en production (best_model_efficientNet_finetune.pth) est un
# EfficientNet B0 torchvision ; l'entraînement doit produire la même architecture.
BRAND_ARCHITECTURE = "efficientnet_b0"
BRAND_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Modèles maison (SimpleYOLO) reconnus par leur nom de fichier
CUSTOM_PLATE_MODELS = ("modelemaison.pt", "YOLO_From_Scratch_LicensePlatev2.pt")

# Repli générique pour la détection de plaques : modèle COCO, qui ne connaît
# pas les plaques. Il ne sert qu'à garder l'application démarrable.
GENERIC_PLATE_FALLBACK = "yolov8n.pt"
GENERIC_VEHICLE_MODEL = "yolov8s.pt"


@dataclass
class BrandBundleRef:
    """Paquet de modèle de marque disponible localement, et son libellé d'origine."""

    directory: Path
    label: str          # ex. « registry:vehicle_brand_model@champion (v3) »


@dataclass
class InferenceModels:
    """Les modèles chargés, derrière leurs interfaces (ports)."""

    plate_detector: Optional[Detector] = None
    plate_reader: Optional[PlateReader] = None
    vehicle_detector: Optional[Detector] = None
    brand_classifier: Optional[BrandClassifier] = None
    device: str = "cpu"
    # Origine de chaque modèle (chemin ou nom), pour le statut et les logs
    sources: Dict[str, str] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_model_path(candidates: Sequence[Optional[str]]) -> Optional[Path]:
    """Premier chemin existant parmi les candidats (relatif au répertoire courant ou au projet)."""
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        for option in (path, PROJECT_ROOT / path):
            if option.exists():
                return option
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEURS
# ═══════════════════════════════════════════════════════════════════════════════

def load_plate_detector(path: Optional[str], device: str = "cpu") -> Optional[Detector]:
    """Charge le détecteur de plaques depuis `path` (modèle ultralytics ou SimpleYOLO)."""
    if not path:
        return None
    name = Path(path).name
    try:
        if name in CUSTOM_PLATE_MODELS:
            from src.models.custom_yolo import load_custom_model

            return SimpleYoloDetector(load_custom_model(str(path), device=device), device=device)
        from ultralytics import YOLO

        return UltralyticsDetector(YOLO(str(path)))
    except Exception as exc:  # noqa: BLE001 - un modèle absent ne doit pas bloquer l'application
        logger.error("Détecteur de plaques non chargé (%s) : %s", path, exc)
        return None


def load_plate_reader(model_name: str) -> Optional[PlateReader]:
    try:
        from fast_plate_ocr import LicensePlateRecognizer

        return FastPlateOcrReader(LicensePlateRecognizer(model_name))
    except Exception as exc:  # noqa: BLE001
        logger.error("OCR non chargé (%s) : %s", model_name, exc)
        return None


def load_vehicle_detector(path: str) -> Optional[Detector]:
    try:
        from ultralytics import YOLO

        return UltralyticsDetector(YOLO(str(path)))
    except Exception as exc:  # noqa: BLE001
        logger.error("Détecteur de véhicules non chargé (%s) : %s", path, exc)
        return None


def build_brand_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((BRAND_INPUT_SIZE, BRAND_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_brand_network(num_classes: int):
    """Construit l'architecture du classificateur de marque (sans poids)."""
    import torch
    from torchvision.models import efficientnet_b0

    network = efficientnet_b0(weights=None)
    network.classifier[1] = torch.nn.Linear(network.classifier[1].in_features, num_classes)
    return network


# Paquet de modèle de marque (format du registre) : un dossier contenant
#   weights.pt      les poids seuls (state_dict), chargés avec weights_only=True
#   classes.json    la liste ordonnée des classes
#   manifest.json   le lignage : jeu de données, métriques, origine (facultatif ici)
BUNDLE_WEIGHTS = "weights.pt"
BUNDLE_CLASSES = "classes.json"
BUNDLE_MANIFEST = "manifest.json"


def read_local_checkpoint(path: Path, default_classes: Sequence[str], device: str = "cpu"):
    """
    Lit un checkpoint local historique → (state_dict, classes, infos).

    Formats : dictionnaire {'model_state_dict', 'label_encoder_classes', 'num_classes',
    'epoch', 'val_acc'} ou state_dict brut. Fichier local de confiance : il peut
    contenir des objets Python (weights_only=False). Un modèle venant du registre
    passe, lui, par `read_bundle` (weights_only=True).
    """
    import torch

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    classes = list(default_classes)
    info = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if checkpoint.get("label_encoder_classes") is not None:
            classes = [str(c) for c in checkpoint["label_encoder_classes"]]
        info = {k: checkpoint.get(k) for k in ("epoch", "val_acc", "num_classes") if k in checkpoint}
    else:
        state_dict = checkpoint
    return state_dict, classes, info


def read_bundle(directory: Path, device: str = "cpu"):
    """Lit un paquet du registre → (state_dict, classes, manifeste). Aucun code n'est exécuté."""
    import json

    import torch

    directory = Path(directory)
    classes = json.loads((directory / BUNDLE_CLASSES).read_text(encoding="utf-8"))
    state_dict = torch.load(str(directory / BUNDLE_WEIGHTS), map_location=device, weights_only=True)
    manifest_path = directory / BUNDLE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return state_dict, [str(c) for c in classes], manifest


def classifier_from_state(state_dict, classes: Sequence[str], device: str = "cpu") -> TorchBrandClassifier:
    """Construit le classificateur à partir de poids et de classes (architecture unique du projet)."""
    weight = state_dict.get("classifier.1.weight")
    num_classes = int(weight.shape[0]) if weight is not None else len(classes)
    if num_classes != len(classes):
        raise ValueError(f"{num_classes} sorties pour {len(classes)} classes : paquet incohérent")
    network = build_brand_network(num_classes)
    network.load_state_dict(state_dict)
    network.to(device).eval()
    return TorchBrandClassifier(network, build_brand_transform(), classes, device=device)


def load_brand_classifier(
    path: Optional[Path], default_classes: Sequence[str], device: str = "cpu"
) -> Optional[BrandClassifier]:
    """Charge le classificateur de marque depuis un checkpoint local historique."""
    if path is None:
        return None
    try:
        state_dict, classes, info = read_local_checkpoint(path, default_classes, device)
        if info:
            logger.info("Checkpoint de marque : epoch=%s, val_acc=%s", info.get("epoch"), info.get("val_acc", "N/A"))
        return classifier_from_state(state_dict, classes, device)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Classificateur de marque non chargé (%s) — architecture attendue : %s : %s",
            path, BRAND_ARCHITECTURE, exc,
        )
        return None


def load_brand_bundle(directory: Optional[Path], device: str = "cpu") -> Optional[BrandClassifier]:
    """Charge le classificateur depuis un paquet du registre (téléchargé localement)."""
    if directory is None:
        return None
    try:
        state_dict, classes, _ = read_bundle(directory, device)
        return classifier_from_state(state_dict, classes, device)
    except Exception as exc:  # noqa: BLE001
        logger.error("Paquet de modèle de marque illisible (%s) : %s", directory, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

def load_models(
    settings: Optional[Settings] = None,
    load_branch2: bool = True,
    plate_model_path: Optional[str] = None,
    brand_bundle: Optional["BrandBundleRef"] = None,
) -> InferenceModels:
    """
    Charge tous les modèles selon la configuration.

    `brand_bundle` : paquet du registre déjà téléchargé (fourni par la couche MLOps).
    S'il est absent ou illisible, le classificateur vient du fichier local.
    """
    settings = settings or get_settings()
    cfg = settings.models
    device = detect_device()
    models = InferenceModels(device=device)

    # Branche 1 : plaque + OCR
    plate_path = resolve_model_path([plate_model_path, cfg.yolo_plate_path, "models/best.pt"])
    if plate_path is None:
        logger.warning(
            "Aucun modèle de plaques trouvé (%s) : repli sur %s, modèle générique "
            "qui ne détecte PAS les plaques.", cfg.yolo_plate_path, GENERIC_PLATE_FALLBACK,
        )
    plate_source = str(plate_path) if plate_path else GENERIC_PLATE_FALLBACK
    models.plate_detector = load_plate_detector(plate_source, device)
    models.sources["plate_detector"] = plate_source

    models.plate_reader = load_plate_reader(cfg.ocr_model)
    models.sources["plate_reader"] = cfg.ocr_model

    if load_branch2:
        # Branche 2 : véhicule + marque
        vehicle_path = resolve_model_path([cfg.yolo_vehicle_path])
        vehicle_source = str(vehicle_path) if vehicle_path else GENERIC_VEHICLE_MODEL
        models.vehicle_detector = load_vehicle_detector(vehicle_source)
        models.sources["vehicle_detector"] = vehicle_source

        if brand_bundle is not None:
            models.brand_classifier = load_brand_bundle(brand_bundle.directory, device)
            models.sources["brand_classifier"] = brand_bundle.label
        if models.brand_classifier is None:
            if brand_bundle is not None:
                logger.warning("Modèle du registre inutilisable : repli sur le fichier local.")
            brand_path = resolve_model_path([cfg.efficientnet_path])
            if brand_path is None:
                logger.warning("%s introuvable : classification de marque désactivée.", cfg.efficientnet_path)
            models.brand_classifier = load_brand_classifier(brand_path, cfg.brand_classes, device)
            models.sources["brand_classifier"] = f"local:{brand_path}" if brand_path else "absent"

    for role, source in models.sources.items():
        loaded = getattr(models, role) is not None
        logger.info("%s %s : %s", "✅" if loaded else "⚠️", role, source)
    return models
