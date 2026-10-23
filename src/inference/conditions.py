"""Estimation des conditions de prise de vue (luminosité, netteté)."""

from __future__ import annotations

import cv2
import numpy as np

from .types import ImageConditions

# Seuils de luminosité moyenne (niveaux de gris 0-255)
_LIGHTING_LEVELS = (
    (60, "Night / Very low light", "🌙"),
    (100, "Low light", "🌆"),
    (160, "Medium light", "☁️"),
)
_DAYLIGHT = ("Daylight / Well lit", "☀️")

# Seuils de netteté (variance du laplacien)
_BLUR_HIGH = 100
_BLUR_MEDIUM = 500


def estimate_conditions(image_bgr: np.ndarray) -> ImageConditions:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    brightness = float(np.mean(gray))

    lighting, emoji = _DAYLIGHT
    for limit, label, icon in _LIGHTING_LEVELS:
        if brightness < limit:
            lighting, emoji = label, icon
            break

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < _BLUR_HIGH:
        blur = "High blur"
    elif blur_score < _BLUR_MEDIUM:
        blur = "Medium blur"
    else:
        blur = "Sharp"

    return ImageConditions(
        brightness=brightness,
        lighting=lighting,
        lighting_emoji=emoji,
        blur=blur,
        blur_score=blur_score,
    )
