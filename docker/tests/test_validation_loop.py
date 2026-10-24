"""
Test de bout en bout de la boucle de validation, avec de vrais composants :
base SQLite, stockage local et détecteur d'incohérences (modèles factices).

    passage incohérent → image sauvegardée → file de validation
        → décision de l'opérateur → image dans le jeu de données
"""

import sqlite3
from pathlib import Path

from src.data.database import DatabaseManager
from src.data.storage import StorageManager
from src.models.mismatch_detector import MismatchDetector
from src.services import ALPRService
from src.services.validation import REJECT, VALIDATE_DECLARED, ValidationService
from tests.test_inference_core import make_engine, make_image

RESIDENT = {
    "plaque": "AB123CD", "nom": "Martin", "prenom": "Léa", "age": 40, "telephone": "0600000000",
    "adresse": "1 rue de la Paix", "ville": "Paris", "code_postal": "75001",
    "abonnement": "non", "acces": "oui", "marque_declaree": "Peugeot",   # le modèle prédit « renault »
}


def make_loop(tmp_path: Path):
    db = DatabaseManager(db_path=str(tmp_path / "alpr.db"))
    db.add_resident(RESIDENT)

    storage = StorageManager()
    storage.mode, storage.s3_client = "local", None
    storage.local_paths = {name: tmp_path / name for name in ("dataset", "captures", "artifacts")}
    for path in storage.local_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    detector = MismatchDetector(db=db, storage=storage)
    service = ALPRService(make_engine(), db, mismatch_detector=detector)
    return db, storage, detector, service


def test_validated_mismatch_reaches_dataset(tmp_path):
    db, storage, detector, service = make_loop(tmp_path)

    # 1. Passage incohérent : l'image du véhicule est sauvegardée et reliée au passage
    outcome = service.process(make_image())
    assert outcome.mismatch is True
    assert outcome.mismatch_image_key is not None
    assert (storage.local_paths["captures"] / outcome.mismatch_image_key).exists()

    # 2. La file de validation (lue en base) connaît l'image
    pending = detector.get_pending()
    assert len(pending) == 1
    assert pending[0].image_path == outcome.mismatch_image_key

    assert detector.get_stats()["queue_size"] == 1                     # compteur « En attente »

    # 3. L'opérateur valide : la base avait raison
    message = ValidationService(detector).act("AB123CD", VALIDATE_DECLARED)
    assert detector.get_stats()["queue_size"] == 0                     # le compteur suit la décision
    assert "image ajoutée au jeu de données" in message

    # 4. L'image est dans le jeu de données, rangée sous la bonne marque
    labeled = list((storage.local_paths["dataset"] / "labeled" / "peugeot").glob("*.jpg"))
    assert len(labeled) == 1

    # 5. La décision est en base : la file est vide, même après un « redémarrage »
    assert detector.get_pending() == []
    assert MismatchDetector(db=db, storage=storage).get_pending() == []
    log = db.get_logs(limit=1)[0]
    assert log["validation_status"] == "validated" and log["validated_label"] == "peugeot"


def test_rejected_mismatch_leaves_queue_without_dataset(tmp_path):
    db, storage, detector, service = make_loop(tmp_path)
    service.process(make_image())

    assert "Rejeté" in ValidationService(detector).act("AB123CD", REJECT)
    assert detector.get_pending() == []
    assert not (storage.local_paths["dataset"] / "labeled").exists()
    assert db.get_logs(limit=1)[0]["validation_status"] == "rejected"


def test_existing_database_is_migrated(tmp_path):
    """Une base créée avant la boucle de validation reçoit les nouvelles colonnes, sans perte."""
    path = tmp_path / "ancienne.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE logs (id INTEGER PRIMARY KEY AUTOINCREMENT, plaque TEXT NOT NULL,
                        timestamp TEXT NOT NULL, resultat TEXT NOT NULL, normalized_plate TEXT NOT NULL,
                        marque_predite TEXT, marque_confiance REAL, mismatch INTEGER DEFAULT 0)""")
        conn.execute("""INSERT INTO logs (plaque, timestamp, resultat, normalized_plate, mismatch)
                        VALUES ('AB123CD', '2026-10-22T10:00:00', 'autorisé', 'AB123CD', 1)""")

    db = DatabaseManager(db_path=str(path))
    log = db.get_logs(limit=1)[0]
    assert log["plaque"] == "AB123CD"
    assert {"image_key", "validation_status", "validated_label", "validated_at"} <= set(log)
    assert len(db.get_mismatch_logs()) == 1          # l'ancienne incohérence reste en attente
