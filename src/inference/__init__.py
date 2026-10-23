"""
Cœur d'inférence ALPR.

Unique implémentation de l'analyse d'une image, utilisée par tous les points
d'entrée (interface Gradio, ligne de commande, Ray Serve).

    from src.inference import ALPREngine
    engine = ALPREngine()
    result = engine.analyze("voiture.jpg")
    print(result.to_dict())
"""

from .brands import is_brand_mismatch, normalize_brand
from .engine import ALPREngine, load_image
from .loader import BRAND_ARCHITECTURE, BRAND_INPUT_SIZE, InferenceModels, load_models
from .types import BrandResult, Detection, ImageConditions, InferenceResult, PlateResult, VehicleResult

__all__ = [
    "ALPREngine",
    "BRAND_ARCHITECTURE",
    "BRAND_INPUT_SIZE",
    "BrandResult",
    "Detection",
    "ImageConditions",
    "InferenceModels",
    "InferenceResult",
    "PlateResult",
    "VehicleResult",
    "is_brand_mismatch",
    "load_image",
    "load_models",
    "normalize_brand",
]
