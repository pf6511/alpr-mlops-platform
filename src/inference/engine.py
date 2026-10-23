"""
Moteur d'inférence ALPR : la seule implémentation de l'analyse d'une image.

Branche 1 : détection de plaque → OCR.
Branche 2 : détection du véhicule → classification de la marque.

Le moteur ne connaît que les interfaces des modèles (voir `models.py`) et ne
fait ni affichage, ni accès base de données : il renvoie un `InferenceResult`.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from configs.settings import Settings, get_settings

from .conditions import estimate_conditions
from .loader import InferenceModels, load_models, load_plate_detector, resolve_model_path
from .types import BrandResult, InferenceResult, PlateResult, VehicleResult

logger = logging.getLogger(__name__)

# Classes COCO retenues comme véhicules
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

ImageInput = Union[str, Path, np.ndarray]


def load_image(image: ImageInput) -> np.ndarray:
    """Image BGR depuis un chemin ou un tableau numpy (copié)."""
    if isinstance(image, np.ndarray):
        return image.copy()
    loaded = cv2.imread(str(image))
    if loaded is None:
        raise ValueError(f"Impossible de charger l'image : {image}")
    return loaded


def _crop(image: np.ndarray, bbox) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop.copy() if crop.size else None


class ALPREngine:
    """Analyse d'images de véhicules : plaques, véhicule, marque."""

    def __init__(
        self,
        models: Optional[InferenceModels] = None,
        settings: Optional[Settings] = None,
        load_branch2: bool = True,
        plate_model_path: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.models = models or load_models(
            self.settings, load_branch2=load_branch2, plate_model_path=plate_model_path
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ÉTAPES UNITAIRES
    # ═══════════════════════════════════════════════════════════════════════════

    def detect_plates(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[PlateResult]:
        detector = self.models.plate_detector
        if detector is None:
            return []
        threshold = self.settings.models.detection_threshold if conf_threshold is None else conf_threshold
        try:
            detections = detector.detect(image, threshold)
        except Exception as exc:  # noqa: BLE001 - une image ne doit pas faire tomber le service
            logger.warning("Erreur de détection de plaque : %s", exc)
            return []
        return [
            PlateResult(bbox=d.bbox, detection_confidence=d.confidence, crop=_crop(image, d.bbox))
            for d in detections
        ]

    def read_plate(self, plate_crop: Optional[np.ndarray]):
        reader = self.models.plate_reader
        if reader is None or plate_crop is None or plate_crop.size == 0:
            return "", None
        try:
            return reader.read(plate_crop)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Erreur OCR : %s", exc)
            return "", None

    def detect_vehicle(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> Optional[VehicleResult]:
        """Véhicule principal : la plus grande boîte parmi les classes véhicule."""
        detector = self.models.vehicle_detector
        if detector is None:
            return None
        threshold = self.settings.models.vehicle_threshold if conf_threshold is None else conf_threshold
        try:
            detections = detector.detect(image, threshold)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Erreur de détection de véhicule : %s", exc)
            return None

        def area(d):
            x1, y1, x2, y2 = d.bbox
            return (x2 - x1) * (y2 - y1)

        vehicles = [d for d in detections if d.class_id in VEHICLE_CLASSES]
        if not vehicles:
            return None
        best = max(vehicles, key=area)
        return VehicleResult(
            bbox=best.bbox,
            class_name=VEHICLE_CLASSES[best.class_id],
            confidence=best.confidence,
            crop=_crop(image, best.bbox),
        )

    def classify_brand(self, vehicle_crop: Optional[np.ndarray]) -> Optional[BrandResult]:
        classifier = self.models.brand_classifier
        if classifier is None or vehicle_crop is None or vehicle_crop.size == 0:
            return None
        try:
            top = classifier.predict(vehicle_crop, top_k=3)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Erreur de classification de marque : %s", exc)
            return None
        if not top:
            return None
        return BrandResult(brand=top[0][0], confidence=top[0][1], top3=top)

    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSE COMPLÈTE
    # ═══════════════════════════════════════════════════════════════════════════

    def analyze(
        self,
        image: ImageInput,
        conf_threshold: Optional[float] = None,
        run_branch2: bool = True,
    ) -> InferenceResult:
        """Analyse complète d'une image (BGR ou chemin)."""
        start = time.perf_counter()
        img = load_image(image)

        plates = self.detect_plates(img, conf_threshold)
        for plate in plates:
            plate.text, plate.ocr_confidence = self.read_plate(plate.crop)

        vehicle = brand = None
        if run_branch2:
            vehicle = self.detect_vehicle(img)
            if vehicle is not None:
                brand = self.classify_brand(vehicle.crop)

        return InferenceResult(
            image_size=(int(img.shape[0]), int(img.shape[1])),
            timestamp=datetime.now().isoformat(),
            inference_time_s=time.perf_counter() - start,
            conditions=estimate_conditions(img),
            plates=plates,
            vehicle=vehicle,
            brand=brand,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # GESTION DES MODÈLES
    # ═══════════════════════════════════════════════════════════════════════════

    def reload_plate_detector(self, model_name: str):
        """Remplace le détecteur de plaques par un fichier de models/. Renvoie (succès, message)."""
        path = resolve_model_path([model_name, f"models/{Path(model_name).name}"])
        if path is None:
            return False, f"❌ Modèle non trouvé : {model_name}"
        detector = load_plate_detector(str(path), self.models.device)
        if detector is None:
            return False, f"❌ Échec du chargement : {model_name}"
        self.models.plate_detector = detector
        self.models.sources["plate_detector"] = str(path)
        return True, f"✅ Modèle '{Path(model_name).name}' chargé !"

    def status(self) -> dict:
        """État des modèles : chargé ou non, et origine."""
        roles = ("plate_detector", "plate_reader", "vehicle_detector", "brand_classifier")
        return {
            role: {
                "loaded": getattr(self.models, role) is not None,
                "source": self.models.sources.get(role, "non configuré"),
            }
            for role in roles
        } | {"device": self.models.device}
