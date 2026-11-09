"""
Tests de l'API REST (src/api/main.py).

Mêmes doublures que les tests des services : faux moteur, fausse base. Aucun modèle,
aucune base réelle. On vérifie le contrat (codes, champs) et la sécurité.
"""

from types import SimpleNamespace

import cv2
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")      # python-multipart : envoi de fichiers (formulaires)

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import create_app  # noqa: E402
from src.services import ALPRService  # noqa: E402
from tests.test_inference_core import make_engine, make_image  # noqa: E402
from tests.test_services import RESIDENT_BLOCKED, RESIDENT_OK, FakeDB, FakeMismatchDetector  # noqa: E402

KEY = "cle-de-test"


def make_client(residents=None, engine=None, **kwargs):
    """Client de test : vrai service métier, faux moteur et fausse base."""
    engine = engine or make_engine()
    db = FakeDB(residents)
    services = SimpleNamespace(
        capture=ALPRService(engine, db, mismatch_detector=FakeMismatchDetector()),
        system=SimpleNamespace(engine=engine),
    )
    app = create_app(services=services, api_key=KEY, **kwargs)
    return TestClient(app), db


def png_bytes(image=None):
    ok, buffer = cv2.imencode(".png", make_image() if image is None else image)
    return buffer.tobytes()


def post_access(client, data=None, key=KEY, content_type="image/png"):
    files = {"image": ("car.png", png_bytes() if data is None else data, content_type)}
    headers = {"X-API-Key": key} if key is not None else {}
    return client.post("/v1/access", files=files, data={"gate_id": "entree-nord"}, headers=headers)


# ═══════════════════════════════════════════════════════════════════════════════
# SÉCURITÉ
# ═══════════════════════════════════════════════════════════════════════════════

def test_refuses_to_start_without_key():
    services = SimpleNamespace(capture=None, system=None)
    with pytest.raises(RuntimeError):
        with TestClient(create_app(services=services, api_key="")):
            pass


def test_access_requires_key():
    client, db = make_client({"AB123CD": RESIDENT_OK})
    with client:
        assert post_access(client, key=None).status_code == 401
        response = post_access(client, key="mauvaise-cle")
    assert response.status_code == 401
    assert response.json()["error"] == "UNAUTHORIZED" and response.json()["request_id"]
    assert response.headers["X-Request-ID"] == response.json()["request_id"]   # même identifiant
    assert db.logs == []                              # rien n'est traité sans clé


def test_health_is_open():
    client, _ = make_client()
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok" and response.json()["models_ready"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRÔLES D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

def test_rejects_too_large_image():
    client, _ = make_client(max_image_mb=0.0001)      # environ 100 octets
    with client:
        response = post_access(client)
    assert response.status_code == 413 and response.json()["error"] == "IMAGE_TOO_LARGE"


def test_rejects_unsupported_format():
    client, _ = make_client()
    with client:
        assert post_access(client, content_type="image/gif").status_code == 415


def test_rejects_unreadable_image():
    client, _ = make_client()
    with client:
        response = post_access(client, data=b"pas une image")
    assert response.status_code == 400 and response.json()["error"] == "INVALID_IMAGE"


# ═══════════════════════════════════════════════════════════════════════════════
# DÉCISIONS (contrat de la réponse)
# ═══════════════════════════════════════════════════════════════════════════════

def test_access_authorized_and_logged():
    client, db = make_client({"AB123CD": RESIDENT_OK})
    with client:
        response = post_access(client)
    body = response.json()
    assert response.status_code == 200
    assert body["plate"] == "AB123CD"
    assert body["access"] == {"authorized": True, "reason": "AUTHORIZED"}
    assert body["brand"]["predicted"] == "renault" and body["mismatch"] is False
    assert len(db.logs) == 1                          # même journalisation que l'interface
    assert "Martin" not in str(body)                  # minimisation : pas d'identité du résident


def test_access_unknown_plate():
    client, _ = make_client({})
    with client:
        body = post_access(client).json()
    assert body["access"] == {"authorized": False, "reason": "UNKNOWN_PLATE"}


def test_access_revoked():
    client, _ = make_client({"AB123CD": RESIDENT_BLOCKED})
    with client:
        body = post_access(client).json()
    assert body["access"] == {"authorized": False, "reason": "ACCESS_REVOKED"}


def test_no_plate():
    from tests.test_inference_core import FakeDetector

    client, db = make_client(engine=make_engine(plate_detector=FakeDetector([])))
    with client:
        body = post_access(client).json()
    assert body["plate"] is None
    assert body["access"] == {"authorized": False, "reason": "NO_PLATE"}
    assert db.logs == []                              # pas de plaque : pas de passage journalisé
