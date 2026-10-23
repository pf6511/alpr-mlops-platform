"""
Ports et adaptateurs des modèles.

Le moteur d'inférence ne dépend que des trois interfaces (ports) ci-dessous.
Les adaptateurs encapsulent les bibliothèques concrètes (ultralytics,
fast-plate-ocr, torchvision) : c'est le seul module qui les manipule.
Cette séparation permet de tester le moteur avec de faux modèles, et de
changer d'implémentation (registre MLflow, ONNX...) sans toucher au moteur.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from .types import Detection

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PORTS
# ═══════════════════════════════════════════════════════════════════════════════

class Detector(Protocol):
    """Détecteur d'objets : image BGR → boîtes au-dessus du seuil."""

    def detect(self, image_bgr: np.ndarray, conf_threshold: float) -> List[Detection]:
        ...


class PlateReader(Protocol):
    """OCR de plaque : image BGR recadrée → (texte, confiance ou None)."""

    def read(self, plate_bgr: np.ndarray) -> Tuple[str, Optional[float]]:
        ...


class BrandClassifier(Protocol):
    """Classificateur de marque : image BGR du véhicule → [(marque, proba)] triées."""

    def predict(self, vehicle_bgr: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTATEURS : DÉTECTION
# ═══════════════════════════════════════════════════════════════════════════════

class UltralyticsDetector:
    """Adaptateur pour un modèle YOLO ultralytics."""

    def __init__(self, model: Any):
        self.model = model

    def detect(self, image_bgr: np.ndarray, conf_threshold: float) -> List[Detection]:
        detections: List[Detection] = []
        for result in self.model(image_bgr, conf=conf_threshold, verbose=False):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                    )
                )
        return detections


class SimpleYoloDetector:
    """Adaptateur pour le modèle maison SimpleYOLO (grille 13×13, entrée 416×416)."""

    INPUT_SIZE = 416
    GRID_SIZE = 13

    def __init__(self, model: Any, device: str = "cpu"):
        self.model = model
        self.device = device

    def detect(self, image_bgr: np.ndarray, conf_threshold: float) -> List[Detection]:
        import torch

        img_h, img_w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, (self.INPUT_SIZE, self.INPUT_SIZE))
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)[0].cpu()

        detections: List[Detection] = []
        g = self.GRID_SIZE
        for cy in range(g):
            for cx in range(g):
                obj_conf = pred[cy, cx, 0].item()
                if obj_conf <= conf_threshold:
                    continue
                xc = (cx + pred[cy, cx, 1].item()) / g
                yc = (cy + pred[cy, cx, 2].item()) / g
                w = abs(pred[cy, cx, 3].item())
                h = abs(pred[cy, cx, 4].item())
                x1 = max(0, int((xc - w / 2) * img_w))
                y1 = max(0, int((yc - h / 2) * img_h))
                x2 = min(img_w, int((xc + w / 2) * img_w))
                y2 = min(img_h, int((yc + h / 2) * img_h))
                if x2 > x1 and y2 > y1:
                    detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=obj_conf))
        return detections


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTATEUR : OCR
# ═══════════════════════════════════════════════════════════════════════════════

def _mean_probability(probs: Any) -> Optional[float]:
    """Moyenne des probabilités par caractère, ou None si absentes/invalides."""
    if probs is None:
        return None
    try:
        values = np.asarray(probs, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return float(values.mean()) if values.size else None


def parse_ocr_output(output: Any) -> Tuple[str, Optional[float]]:
    """
    Normalise la sortie de `LicensePlateRecognizer.run`.

    Deux formats coexistent selon la version de fast-plate-ocr :
    - récent : liste d'objets prédiction (attribut `plate`, probabilités par
      caractère éventuelles) ;
    - ancien, avec `return_confidence=True` : tuple `(textes, probabilités)`.
    """
    if not output:
        return "", None

    # Ancien format : (liste de textes, tableau de probabilités)
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], (list, tuple)):
        texts, probs = output
        if not texts:
            return "", None
        first_probs = probs[0] if probs is not None and len(probs) else None
        return str(texts[0]), _mean_probability(first_probs)

    first = output[0]
    if hasattr(first, "plate"):
        probs = None
        for attr in ("char_probs", "probs", "confidence"):
            if getattr(first, attr, None) is not None:
                probs = getattr(first, attr)
                break
        return str(first.plate), _mean_probability(probs)

    return str(first), None


def normalize_plate_text(text: str) -> str:
    """Majuscules, sans espaces ni tirets, sans caractère de remplissage."""
    return text.upper().replace(" ", "").replace("-", "").replace("_", "")


class FastPlateOcrReader:
    """Adaptateur pour fast-plate-ocr (entrée en niveaux de gris, validée en test)."""

    def __init__(self, recognizer: Any):
        self.recognizer = recognizer

    def read(self, plate_bgr: np.ndarray) -> Tuple[str, Optional[float]]:
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if plate_bgr.ndim == 3 else plate_bgr
        try:
            output = self.recognizer.run(gray, return_confidence=True)
        except TypeError:
            # Version sans paramètre return_confidence
            output = self.recognizer.run(gray)
        text, confidence = parse_ocr_output(output)
        return normalize_plate_text(text), confidence


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTATEUR : CLASSIFICATION DE MARQUE
# ═══════════════════════════════════════════════════════════════════════════════

class TorchBrandClassifier:
    """Adaptateur pour un classificateur PyTorch (EfficientNet B0 torchvision)."""

    def __init__(self, model: Any, transform: Any, classes: Sequence[str], device: str = "cpu"):
        self.model = model
        self.transform = transform
        self.classes = list(classes)
        self.device = device

    def _label(self, index: int) -> str:
        return self.classes[index] if index < len(self.classes) else f"class_{index}"

    def predict(self, vehicle_bgr: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        rgb = cv2.cvtColor(vehicle_bgr, cv2.COLOR_BGR2RGB) if vehicle_bgr.ndim == 3 else vehicle_bgr
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1)[0]
        values, indices = torch.topk(probs, min(top_k, probs.shape[0]))
        return [(self._label(i), float(p)) for i, p in zip(indices.tolist(), values.tolist())]
