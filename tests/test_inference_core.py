"""
Tests du cœur d'inférence (src/inference) et de ses trois adaptateurs.

Les modèles sont remplacés par des doublures qui respectent les interfaces
(ports) : ces tests ne dépendent ni de torch, ni des poids, ni du matériel.
"""

import base64
import json
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from configs.settings import get_settings
from src.inference import (
    BRAND_ARCHITECTURE,
    ALPREngine,
    Detection,
    InferenceModels,
    is_brand_mismatch,
    normalize_brand,
)
from src.inference.models import normalize_plate_text, parse_ocr_output

# ═══════════════════════════════════════════════════════════════════════════════
# DOUBLURES DE MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

PLATE_BOX = (40, 120, 140, 150)
CAR_BOX = (10, 20, 300, 200)        # la plus grande boîte de classe véhicule
SMALL_CAR_BOX = (200, 10, 260, 60)
PERSON_BOX = (0, 0, 319, 239)       # plus grande, mais pas un véhicule


@dataclass
class FakeDetector:
    detections: list
    fail: bool = False

    def __post_init__(self):
        self.thresholds = []

    def detect(self, image_bgr, conf_threshold):
        self.thresholds.append(conf_threshold)
        if self.fail:
            raise RuntimeError("panne simulée")
        return list(self.detections)


class FakeReader:
    def __init__(self, text="AB123CD", confidence=0.91):
        self.text, self.confidence = text, confidence

    def read(self, plate_bgr):
        return self.text, self.confidence


class FakeClassifier:
    def predict(self, vehicle_bgr, top_k=3):
        return [("renault", 0.82), ("dacia", 0.10), ("peugeot", 0.05)][:top_k]


def make_image():
    """Image BGR 320×240 ; le véhicule contient un pixel bleu pur (BGR) en (25, 15)."""
    img = np.full((240, 320, 3), 127, dtype=np.uint8)
    img[25, 15] = (255, 0, 0)
    return img


def make_engine(**overrides):
    models = InferenceModels(
        plate_detector=FakeDetector([Detection(PLATE_BOX, 0.88)]),
        plate_reader=FakeReader(),
        vehicle_detector=FakeDetector([
            Detection(PERSON_BOX, 0.95, class_id=0),
            Detection(SMALL_CAR_BOX, 0.70, class_id=2),
            Detection(CAR_BOX, 0.90, class_id=2),
        ]),
        brand_classifier=FakeClassifier(),
        sources={"plate_detector": "fake"},
    )
    for name, value in overrides.items():
        setattr(models, name, value)
    return ALPREngine(models=models)


# ═══════════════════════════════════════════════════════════════════════════════
# MOTEUR
# ═══════════════════════════════════════════════════════════════════════════════

def test_analyze_full_pipeline():
    result = make_engine().analyze(make_image())

    assert len(result.plates) == 1
    plate = result.plates[0]
    assert plate.bbox == PLATE_BOX
    assert plate.text == "AB123CD"
    assert plate.ocr_confidence == pytest.approx(0.91)
    assert plate.crop.shape == (30, 100, 3)

    assert result.vehicle.bbox == CAR_BOX            # plus grande boîte véhicule, pas la personne
    assert result.vehicle.class_name == "car"
    assert result.brand.brand == "renault"
    assert result.brand.top3[0] == ("renault", 0.82)
    assert result.best_plate is plate
    assert result.image_size == (240, 320)


def test_thresholds_come_from_configuration():
    engine = make_engine()
    cfg = get_settings().models
    engine.analyze(make_image())
    assert engine.models.plate_detector.thresholds == [cfg.detection_threshold]
    assert engine.models.vehicle_detector.thresholds == [cfg.vehicle_threshold]

    engine.analyze(make_image(), conf_threshold=0.6)
    assert engine.models.plate_detector.thresholds[-1] == 0.6


def test_branch2_can_be_disabled():
    engine = make_engine()
    result = engine.analyze(make_image(), run_branch2=False)
    assert result.vehicle is None and result.brand is None
    assert engine.models.vehicle_detector.thresholds == []


def test_missing_models_degrade_gracefully():
    engine = ALPREngine(models=InferenceModels())
    result = engine.analyze(make_image())
    assert result.plates == [] and result.vehicle is None and result.brand is None


def test_model_failure_does_not_crash():
    engine = make_engine(plate_detector=FakeDetector([], fail=True))
    result = engine.analyze(make_image())
    assert result.plates == []
    assert result.brand.brand == "renault"            # la branche 2 continue


def test_result_is_json_serializable():
    payload = make_engine().analyze(make_image()).to_dict()
    text = json.dumps(payload)
    assert "crop" not in text
    assert payload["brand"]["top3"][0] == {"brand": "renault", "confidence": 0.82}


def test_engine_accepts_image_path(tmp_path):
    path = tmp_path / "car.png"
    cv2.imwrite(str(path), make_image())
    assert make_engine().analyze(str(path)).plates[0].text == "AB123CD"
    with pytest.raises(ValueError):
        make_engine().analyze(str(tmp_path / "absent.png"))


# ═══════════════════════════════════════════════════════════════════════════════
# COHÉRENCE DES TROIS POINTS D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_entry_points_agree():
    """Même image, même moteur : Gradio, CLI et Ray Serve renvoient le même résultat."""
    from pipelines.inference_pipeline import InferencePipeline
    from src.models.pipeline import ALPRPipeline
    from src.serving.ray_serve import handle_process

    engine = make_engine()
    image = make_image()

    gradio = ALPRPipeline(engine=engine).process_image(image, declared_brand="Peugeot")
    cli = InferencePipeline(engine=engine).process(image, declared_brand="Peugeot")
    ok, png = cv2.imencode(".png", image)
    ray = handle_process(engine, {"image": base64.b64encode(png).decode(), "declared_brand": "Peugeot"})

    assert gradio["step4_ocr"][0]["text"] == cli["plates"][0]["text"] == ray["plates"][0]["text"] == "AB123CD"
    assert list(gradio["step4_ocr"][0]["bbox"]) == cli["plates"][0]["bbox"] == ray["plates"][0]["bbox"]
    assert gradio["brand_result"]["brand"] == cli["brand"]["brand"] == ray["brand"]["brand"] == "renault"
    assert gradio["vehicle_detection"]["bbox"] == cli["vehicle"]["bbox"] == ray["vehicle"]["bbox"]
    assert gradio["mismatch"] is cli["mismatch"] is ray["mismatch"] is True


def test_gradio_adapter_returns_rgb_images():
    from src.models.pipeline import ALPRPipeline

    result = ALPRPipeline(engine=make_engine()).process_image(make_image())
    crop = result["vehicle_crop"]                     # recadrage commençant en (10, 20)
    assert tuple(crop[25 - 20, 15 - 10]) == (0, 0, 255)   # bleu BGR → bleu en RGB
    assert result["step1_raw"].shape == (240, 320, 3)
    assert result["step2_detection"] is not None and result["step5_final"] is not None


def test_ray_adapter_validates_input():
    from src.serving.ray_serve import handle_process

    with pytest.raises(ValueError):
        handle_process(make_engine(), {})
    with pytest.raises(ValueError):
        handle_process(make_engine(), {"image": base64.b64encode(b"pas une image").decode()})


def test_ray_adapter_returns_annotated_image():
    from src.serving.ray_serve import decode_image, handle_process

    ok, png = cv2.imencode(".png", make_image())
    response = handle_process(make_engine(), {"image": base64.b64encode(png).decode(), "return_images": True})
    assert decode_image(response["annotated_image"]).shape == (240, 320, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# OCR ET MARQUES
# ═══════════════════════════════════════════════════════════════════════════════

def test_ocr_output_formats():
    recent = [SimpleNamespace(plate="ab-123_cd", char_probs=[0.9, 0.7, 0.8])]
    text, conf = parse_ocr_output(recent)
    assert normalize_plate_text(text) == "AB123CD"
    assert conf == pytest.approx(0.8)

    legacy = (["AB123CD"], np.array([[0.6, 0.8]]))
    assert parse_ocr_output(legacy) == ("AB123CD", pytest.approx(0.7))

    assert parse_ocr_output([SimpleNamespace(plate="AB123CD")]) == ("AB123CD", None)
    assert parse_ocr_output([]) == ("", None)


def test_brand_rules():
    assert normalize_brand(" VW ") == "volkswagen"
    assert normalize_brand("Mercedes-Benz") == "mercedes"
    assert normalize_brand("Citroën") == "citroen"
    assert is_brand_mismatch("renault", "Peugeot") is True
    assert is_brand_mismatch("volkswagen", "VW") is False
    assert is_brand_mismatch("renault", None) is False


def test_mismatch_detector_uses_shared_rule():
    from src.models.mismatch_detector import MismatchDetector

    detector = MismatchDetector()
    assert detector.check_mismatch("AB123CD", "citroen", "Citroën") is False
    assert detector.check_mismatch("AB123CD", "renault", "peugeot") is True


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE DU CLASSIFICATEUR DE MARQUE
# ═══════════════════════════════════════════════════════════════════════════════

def test_brand_network_is_efficientnet_b0():
    """Le checkpoint en production est un B0 torchvision : toute divergence doit casser ce test."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.inference.loader import BRAND_INPUT_SIZE, build_brand_network

    assert BRAND_ARCHITECTURE == "efficientnet_b0"
    network = build_brand_network(22).eval()
    assert network.classifier[1].in_features == 1280      # signature d'EfficientNet B0
    with torch.no_grad():
        logits = network(torch.zeros(1, 3, BRAND_INPUT_SIZE, BRAND_INPUT_SIZE))
    assert logits.shape == (1, 22)
