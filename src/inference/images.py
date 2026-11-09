"""
Conversions d'images communes aux points d'entrée « machine » (API FastAPI, Ray Serve).

Le moteur travaille sur des images numpy BGR (convention OpenCV). Ces fonctions
convertissent ce que reçoit un point d'entrée (octets d'un fichier, base64) vers ce
format, et inversement.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Octets d'un fichier image (JPEG, PNG...) → numpy BGR. Lève ValueError si illisible."""
    buffer = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if image is None:
        raise ValueError("Image illisible")
    return image


def decode_image(image_data: str) -> np.ndarray:
    """Image base64 (éventuellement préfixée « data:...;base64, ») → numpy BGR."""
    if "base64," in image_data:
        image_data = image_data.split("base64,", 1)[1]
    return decode_image_bytes(base64.b64decode(image_data))


def encode_image(image_bgr: np.ndarray) -> str:
    """Image numpy BGR → JPEG base64."""
    ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Encodage JPEG impossible")
    return base64.b64encode(buffer).decode("utf-8")
