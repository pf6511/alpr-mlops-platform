"""
Rendu des annotations sur une image.

Séparé du moteur : l'inférence produit des données, l'affichage est l'affaire
des adaptateurs. Les images sont manipulées en RGB, couleurs exprimées en RGB.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import InferenceResult

PLATE_COLOR = (0, 255, 0)        # vert
BRAND_COLOR = (255, 165, 0)      # orange
MISMATCH_COLOR = (255, 0, 0)     # rouge
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_plate_boxes(image_rgb: np.ndarray, result: InferenceResult) -> np.ndarray:
    """Cadres des plaques détectées, sans texte (étape « détection »)."""
    canvas = image_rgb.copy()
    for plate in result.plates:
        x1, y1, x2, y2 = plate.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PLATE_COLOR, 3)
    return canvas


def annotate(image_rgb: np.ndarray, result: InferenceResult, mismatch: bool = False) -> np.ndarray:
    """Image finale : plaques avec leur texte, véhicule avec sa marque."""
    canvas = image_rgb.copy()

    for plate in result.plates:
        x1, y1, x2, y2 = plate.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PLATE_COLOR, 3)
        label = plate.text or "???"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.8, 2)
        cv2.rectangle(canvas, (x1, y1 - th - 10), (x1 + tw + 10, y1), PLATE_COLOR, -1)
        cv2.putText(canvas, label, (x1 + 5, y1 - 5), _FONT, 0.8, (0, 0, 0), 2)

    if result.vehicle is not None and result.brand is not None:
        x1, y1, x2, y2 = result.vehicle.bbox
        color = MISMATCH_COLOR if mismatch else BRAND_COLOR
        label = f"{result.brand.brand} ({result.brand.confidence:.0%})"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, y2 + 25), _FONT, 0.7, color, 2)

    return canvas
