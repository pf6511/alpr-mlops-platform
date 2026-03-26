import pytest
import json
from pathlib import Path
from src.utils.error_gallery import ErrorGallery

ANNOTATED_IMG_FILE_NAME = "error_1770823832.jpg"
IMG_FILE_NAME_NO_ANNOTATION = "img2.jpg"
NO_PLATE_DETECTED = "No Plate Detected"

@pytest.fixture
def gallery_with_data(tmp_path):
    gallery = tmp_path / "error_examples"
    gallery.mkdir()

    
    # images
    (gallery / ANNOTATED_IMG_FILE_NAME).touch()
    (gallery / IMG_FILE_NAME_NO_ANNOTATION).touch()

    # annotations
    annotations = {
        ANNOTATED_IMG_FILE_NAME: {
            "issue": NO_PLATE_DETECTED,
            "expected": "Unknown",
            "predicted": "None",
            "detection_confidence": 0.0,
            "ocr_confidence": 0.0,
            "analysis": "Failed to detect plate. <div class=\"access-banner\" style=\"background-color..."
        }
    }

    with open(gallery / "annotations.json", "w") as f:
        json.dump(annotations, f)

    return gallery

def test_get_examples_with_annotations(gallery_with_data):
    eg = ErrorGallery(gallery_path=str(gallery_with_data))
    results = {Path(p).name: a for p, a in eg.get_examples()}

    assert results[ANNOTATED_IMG_FILE_NAME]["issue"] == NO_PLATE_DETECTED
    assert results[IMG_FILE_NAME_NO_ANNOTATION]["issue"] ==  ErrorGallery.UNKNOWN_ISSUE_VALUE
