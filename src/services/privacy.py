"""
Protection des données (RGPD) : masquer les plaques dans l'image d'un véhicule.

L'image du véhicule (découpe) est conservée en cas d'incohérence, puis copiée dans
le jeu d'entraînement après validation. La plaque y est masquée AVANT tout
enregistrement : l'image seule ne permet plus d'identifier le véhicule.

Méthode : pixellisation à gros blocs. La zone est réduite à quelques pixels, puis
agrandie sans lissage : chaque bloc prend une couleur unie, les caractères sont
illisibles (contrairement à un flou léger, qui peut parfois être inversé).

Géométrie : les boîtes (bbox) sont en coordonnées de l'image ENTIÈRE ; la découpe
du véhicule commence en (vx1, vy1). Dans la découpe, la plaque est donc en
(px1 - vx1, py1 - vy1, px2 - vx1, py2 - vy1), bornée aux dimensions de la découpe.
"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from src.inference.types import BBox

MARGIN = 0.10      # marge autour de la plaque (les boîtes YOLO sont parfois serrées)
BLOCKS = 6         # nombre de blocs sur la largeur de la plaque : 6 blocs = caractères illisibles


def plate_box_in_crop(vehicle_bbox: BBox, plate_bbox: BBox,
                      crop_width: int, crop_height: int) -> Optional[BBox]:
    """Boîte de la plaque dans la découpe du véhicule (marge comprise), ou None si hors découpe."""
    vx1, vy1 = vehicle_bbox[0], vehicle_bbox[1]
    px1, py1, px2, py2 = plate_bbox

    # Marge autour de la plaque
    margin_x = int((px2 - px1) * MARGIN)
    margin_y = int((py2 - py1) * MARGIN)

    # Changement de repère (image entière → découpe), puis bornage à la découpe
    x1 = max(px1 - margin_x - vx1, 0)
    y1 = max(py1 - margin_y - vy1, 0)
    x2 = min(px2 + margin_x - vx1, crop_width)
    y2 = min(py2 + margin_y - vy1, crop_height)

    if x2 <= x1 or y2 <= y1:
        return None                 # la plaque n'est pas dans la découpe
    return (x1, y1, x2, y2)


def pixelate(region: np.ndarray, blocks: int = BLOCKS) -> np.ndarray:
    """Pixellise une zone : réduction à `blocks` pixels de large, puis agrandissement sans lissage."""
    height, width = region.shape[:2]
    small_width = max(1, min(blocks, width))
    small_height = max(1, round(height * small_width / width))
    small = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def mask_plates(vehicle_crop: np.ndarray, vehicle_bbox: BBox,
                plate_bboxes: Iterable[BBox]) -> np.ndarray:
    """Copie de la découpe du véhicule, toutes les plaques qu'elle contient pixellisées."""
    masked = vehicle_crop.copy()            # on ne modifie pas l'image d'origine
    height, width = masked.shape[:2]
    for plate_bbox in plate_bboxes:
        box = plate_box_in_crop(vehicle_bbox, plate_bbox, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        masked[y1:y2, x1:x2] = pixelate(masked[y1:y2, x1:x2])
    return masked
