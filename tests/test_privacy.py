"""
Tests du masquage des plaques (src/services/privacy.py).

Images synthétiques : une « plaque » faite de rayures noires et blanches (comme des
caractères) sur un fond uni. Pixellisée, elle doit perdre ses rayures ; le reste de
l'image doit rester intact.
"""

import numpy as np

from src.services import ALPRService
from src.services.privacy import mask_plates, pixelate, plate_box_in_crop
from tests.test_inference_core import CAR_BOX, PLATE_BOX, make_engine, make_image

VEHICLE_BBOX = (100, 50, 400, 250)          # découpe de 300 × 200, en coordonnées de l'image entière
PLATE_BBOX = (200, 180, 300, 210)           # plaque de 100 × 30, dans la découpe en (100, 130)


def striped_crop():
    """Découpe grise 300 × 200 avec une plaque à rayures verticales de 2 pixels."""
    crop = np.full((200, 300, 3), 127, dtype=np.uint8)
    for x in range(100, 200, 4):
        crop[130:160, x:x + 2] = 255
        crop[130:160, x + 2:x + 4] = 0
    return crop


# ═══════════════════════════════════════════════════════════════════════════════
# GÉOMÉTRIE
# ═══════════════════════════════════════════════════════════════════════════════

def test_plate_box_is_converted_to_crop_coordinates():
    # (200 - 100, 180 - 50, 300 - 100, 210 - 50) = (100, 130, 200, 160), plus 10 % de marge
    assert plate_box_in_crop(VEHICLE_BBOX, PLATE_BBOX, 300, 200) == (90, 127, 210, 163)


def test_plate_box_is_clipped_to_crop():
    """Plaque à cheval sur le bord droit de la découpe : seule la partie visible est gardée."""
    assert plate_box_in_crop(VEHICLE_BBOX, (350, 180, 450, 210), 300, 200) == (240, 127, 300, 163)


def test_plate_outside_crop_is_ignored():
    """Plaque d'un autre véhicule, hors de la découpe."""
    assert plate_box_in_crop(VEHICLE_BBOX, (500, 300, 600, 330), 300, 200) is None


# ═══════════════════════════════════════════════════════════════════════════════
# PIXELLISATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_pixelate_removes_details_and_keeps_size():
    region = striped_crop()[130:160, 100:200]
    result = pixelate(region)
    assert result.shape == region.shape
    # Les rayures (alternance tous les 2 pixels) ont disparu : 6 blocs de ~17 pixels unis
    assert len(np.unique(result[0, :, 0])) <= 6
    assert not np.array_equal(result, region)


def test_mask_plates_changes_only_the_plate():
    crop = striped_crop()
    masked = mask_plates(crop, VEHICLE_BBOX, [PLATE_BBOX])

    x1, y1, x2, y2 = plate_box_in_crop(VEHICLE_BBOX, PLATE_BBOX, 300, 200)
    assert not np.array_equal(masked[y1:y2, x1:x2], crop[y1:y2, x1:x2])    # plaque masquée

    outside = np.ones(crop.shape[:2], dtype=bool)
    outside[y1:y2, x1:x2] = False
    assert np.array_equal(masked[outside], crop[outside])                  # reste intact
    assert crop[130, 100, 0] == 255                                        # l'original n'est pas modifié


def test_mask_plates_without_plate_returns_same_image():
    crop = striped_crop()
    assert np.array_equal(mask_plates(crop, VEHICLE_BBOX, []), crop)
    assert np.array_equal(mask_plates(crop, VEHICLE_BBOX, [(500, 300, 600, 330)]), crop)


# ═══════════════════════════════════════════════════════════════════════════════
# INTÉGRATION : l'image conservée pour validation a sa plaque masquée
# ═══════════════════════════════════════════════════════════════════════════════

class RecordingDetector:
    """Faux MismatchDetector : garde l'image reçue."""

    def detect_and_record(self, vehicle_crop=None, **kwargs):
        self.vehicle_crop = vehicle_crop
        return None


class FakeDB:
    def get_resident_by_plate(self, plate):
        return {"acces": "oui", "marque_declaree": "Peugeot"}     # le moteur factice prédit « renault »

    def add_log(self, **kwargs):
        pass


def test_recorded_vehicle_image_has_plate_masked():
    image = make_image()
    x1, y1, x2, y2 = PLATE_BOX
    for x in range(x1, x2, 4):                                       # rayures sur la plaque
        image[y1:y2, x:x + 2] = 255
        image[y1:y2, x + 2:x + 4] = 0

    detector = RecordingDetector()
    ALPRService(make_engine(), FakeDB(), mismatch_detector=detector).process(image)

    saved = detector.vehicle_crop                                    # RGB, découpe du véhicule
    vx1, vy1 = CAR_BOX[0], CAR_BOX[1]
    plate_in_saved = saved[y1 - vy1:y2 - vy1, x1 - vx1:x2 - vx1]
    plate_in_image = image[y1:y2, x1:x2]
    assert not np.array_equal(plate_in_saved[..., ::-1], plate_in_image)   # plaque masquée
    assert tuple(saved[25 - vy1, 15 - vx1]) == (0, 0, 255)                 # le reste est intact
