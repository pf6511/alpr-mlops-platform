"""
Types de résultat du cœur d'inférence.

Ces structures sont la seule représentation d'un résultat d'analyse : tous les
points d'entrée (Gradio, CLI, Ray Serve) les produisent via le même moteur et ne
diffèrent que par leur mise en forme (`to_dict()` pour le JSON).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    """Sortie brute d'un détecteur d'objets (port `Detector`)."""

    bbox: BBox
    confidence: float
    class_id: int = 0


@dataclass
class PlateResult:
    """Une plaque détectée, et son texte lu par l'OCR."""

    bbox: BBox
    detection_confidence: float
    text: str = ""
    # None quand le moteur OCR ne fournit pas de confiance : on ne l'invente pas.
    ocr_confidence: Optional[float] = None
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "detection_confidence": self.detection_confidence,
            "text": self.text,
            "ocr_confidence": self.ocr_confidence,
        }


@dataclass
class VehicleResult:
    """Le véhicule principal de l'image (le plus grand détecté)."""

    bbox: BBox
    class_name: str
    confidence: float
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "class_name": self.class_name,
            "confidence": self.confidence,
        }


@dataclass
class BrandResult:
    """Prédiction de marque, avec les trois meilleures hypothèses."""

    brand: str
    confidence: float
    top3: List[Tuple[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brand": self.brand,
            "confidence": self.confidence,
            "top3": [{"brand": b, "confidence": c} for b, c in self.top3],
        }


@dataclass
class ImageConditions:
    """Conditions de prise de vue (base d'une future mesure de dérive des données)."""

    brightness: float
    lighting: str
    lighting_emoji: str
    blur: str
    blur_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brightness": self.brightness,
            "lighting": self.lighting,
            "lighting_emoji": self.lighting_emoji,
            "blur": self.blur,
            "blur_score": self.blur_score,
        }


@dataclass
class InferenceResult:
    """Résultat complet de l'analyse d'une image."""

    image_size: Tuple[int, int]
    timestamp: str
    inference_time_s: float
    conditions: ImageConditions
    plates: List[PlateResult] = field(default_factory=list)
    vehicle: Optional[VehicleResult] = None
    brand: Optional[BrandResult] = None

    @property
    def best_plate(self) -> Optional[PlateResult]:
        """Première plaque lue avec un texte non vide (ordre du détecteur)."""
        return next((p for p in self.plates if p.text), None)

    def to_dict(self) -> Dict[str, Any]:
        """Représentation JSON, sans les images recadrées."""
        return {
            "timestamp": self.timestamp,
            "image_size": list(self.image_size),
            "inference_time_s": self.inference_time_s,
            "conditions": self.conditions.to_dict(),
            "plates": [p.to_dict() for p in self.plates],
            "vehicle": self.vehicle.to_dict() if self.vehicle else None,
            "brand": self.brand.to_dict() if self.brand else None,
        }
