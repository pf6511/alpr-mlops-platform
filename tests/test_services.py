"""
Tests de la couche métier (src/services) et de la mise en forme (src/ui/formatters).

Doublures : faux moteur (modèles factices de test_inference_core), fausse base,
faux détecteur d'incohérences. Aucun modèle, aucune base réelle.
"""

import os
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.services import ALPRService, decide_access
from src.services.history import HistoryService
from src.services.residents import ResidentService
from src.services.validation import CUSTOM, REJECT, VALIDATE_DECLARED, VALIDATE_PREDICTED, ValidationService
from tests.test_inference_core import FakeDetector, make_engine, make_image

# ═══════════════════════════════════════════════════════════════════════════════
# DOUBLURES
# ═══════════════════════════════════════════════════════════════════════════════

class FakeDB:
    def __init__(self, residents=None, fail_lookup=False):
        self.residents = residents or {}
        self.fail_lookup = fail_lookup
        self.logs = []
        self.lookups = 0

    def get_resident_by_plate(self, plaque):
        self.lookups += 1
        if self.fail_lookup:
            raise ConnectionError("base indisponible")
        return self.residents.get(plaque)

    def add_log(self, **kwargs):
        self.logs.append(kwargs)
        return True

    def get_logs(self, limit=100):
        return self.logs[:limit]

    def add_resident(self, data):
        return (True, "ok") if data.get("nom") != "ECHEC" else (False, "Plaque déjà enregistrée")

    def get_all_residents(self):
        return list(self.residents.values())


class FakeMismatchDetector:
    def __init__(self, pending=()):
        self.recorded = []
        self.actions = []
        self._pending = list(pending)

    def detect_and_record(self, **kwargs):
        self.recorded.append(kwargs)

    def get_pending(self, limit=50):
        return self._pending[:limit]

    def get_stats(self):
        return {"total_detected": len(self._pending)}

    def validate_as_predicted(self, record):
        self.actions.append(("predicted", record.plaque))
        return True, "validé (modèle)"

    def validate_as_declared(self, record):
        self.actions.append(("declared", record.plaque))
        return True, "validé (base)"

    def reject(self, record, reason):
        self.actions.append(("reject", record.plaque))
        return True, "rejeté"

    def validate(self, record, brand):
        self.actions.append(("custom", record.plaque, brand))
        return True, f"validé ({brand})"


class FakeGallery:
    def __init__(self):
        self.paths = []

    def add_example(self, image_path, **kwargs):
        assert os.path.exists(image_path)          # le fichier existe pendant l'appel
        self.paths.append(image_path)


class CountingEngine:
    """Enveloppe un moteur pour compter les analyses."""

    def __init__(self, engine):
        self.engine, self.calls = engine, 0

    def analyze(self, *args, **kwargs):
        self.calls += 1
        return self.engine.analyze(*args, **kwargs)


RESIDENT_OK = {"plaque": "AB123CD", "acces": "oui", "marque_declaree": "Renault", "nom": "Martin"}
RESIDENT_OTHER_BRAND = {**RESIDENT_OK, "marque_declaree": "Peugeot"}
RESIDENT_BLOCKED = {**RESIDENT_OK, "acces": "non"}


def make_service(residents=None, **kwargs):
    db = FakeDB(residents, fail_lookup=kwargs.pop("fail_lookup", False))
    detector, gallery = FakeMismatchDetector(), FakeGallery()
    engine = CountingEngine(kwargs.pop("engine", make_engine()))
    service = ALPRService(engine, db, mismatch_detector=detector, error_gallery=gallery)
    return service, db, detector, gallery, engine


# ═══════════════════════════════════════════════════════════════════════════════
# ALPRService : un passage
# ═══════════════════════════════════════════════════════════════════════════════

def test_authorized_resident_with_matching_brand():
    service, db, detector, gallery, engine = make_service({"AB123CD": RESIDENT_OK})
    outcome = service.process(make_image())

    assert outcome.plate_text == "AB123CD"
    assert outcome.access.authorized is True
    assert outcome.declared_brand == "Renault"
    assert outcome.mismatch is False
    assert engine.calls == 1                       # une seule analyse
    assert len(db.logs) == 1                       # un seul enregistrement
    assert db.logs[0]["marque_predite"] == "renault" and db.logs[0]["mismatch"] is False
    assert detector.recorded == [] and gallery.paths == []


def test_brand_mismatch_is_logged_once_and_queued():
    service, db, detector, _, engine = make_service({"AB123CD": RESIDENT_OTHER_BRAND})
    outcome = service.process(make_image())

    assert outcome.mismatch is True
    assert engine.calls == 1                       # plus de seconde analyse
    assert len(db.logs) == 1 and db.logs[0]["mismatch"] is True
    assert len(detector.recorded) == 1
    record = detector.recorded[0]
    assert record["log_to_db"] is False            # pas de second enregistrement
    assert record["marque_declaree"] == "Peugeot"
    # Image du véhicule en RGB : le pixel bleu pur (BGR 255,0,0) de make_image
    assert tuple(record["vehicle_crop"][25 - 20, 15 - 10]) == (0, 0, 255)


def test_unknown_plate_is_denied():
    service, db, detector, _, _ = make_service({})
    outcome = service.process(make_image())
    assert outcome.access.authorized is False
    assert outcome.resident is None and outcome.mismatch is False
    assert len(db.logs) == 1 and db.logs[0]["authorized"] is False
    assert detector.recorded == []


def test_resident_without_access_is_denied():
    service, _, _, _, _ = make_service({"AB123CD": RESIDENT_BLOCKED})
    assert service.process(make_image()).access.authorized is False


def test_access_reflects_database_immediately():
    """Un résident ajouté après le démarrage est reconnu tout de suite (plus de cache)."""
    service, db, _, _, _ = make_service({})
    assert service.process(make_image()).access.authorized is False
    db.residents["AB123CD"] = RESIDENT_OK
    assert service.process(make_image()).access.authorized is True


def test_no_plate_records_failure_without_log():
    engine = make_engine(plate_detector=FakeDetector([]))
    service, db, _, gallery, _ = make_service({}, engine=engine)
    outcome = service.process(make_image())

    assert outcome.plate_text == "" and outcome.access is None
    assert db.logs == [] and db.lookups == 0
    assert outcome.failure_recorded is True
    assert len(gallery.paths) == 1 and not os.path.exists(gallery.paths[0])  # temporaire supprimé


def test_brand_disabled_means_no_mismatch():
    service, db, detector, _, _ = make_service({"AB123CD": RESIDENT_OTHER_BRAND})
    outcome = service.process(make_image(), run_brand=False)
    assert outcome.result.brand is None and outcome.mismatch is False
    assert db.logs[0]["marque_predite"] is None and detector.recorded == []


def test_database_failure_does_not_block_analysis():
    service, db, _, _, _ = make_service({}, fail_lookup=True)
    outcome = service.process(make_image())
    assert outcome.plate_text == "AB123CD" and outcome.access.authorized is False


def test_decide_access_rule():
    assert decide_access("X", {"acces": "OUI "}).authorized is True
    assert decide_access("X", {"acces": "non"}).authorized is False
    assert decide_access("X", None).message == "⛔ ACCESS DENIED: X"


# ═══════════════════════════════════════════════════════════════════════════════
# AUTRES SERVICES
# ═══════════════════════════════════════════════════════════════════════════════

def test_resident_service_reports_failures():
    service = ResidentService(FakeDB())
    assert service.add({"plaque": "AB123CD", "nom": "Martin", "prenom": "Léa"})[0] is True
    ok, message = service.add({"plaque": "AB123CD", "nom": "ECHEC"})
    assert ok is False and "déjà" in message        # l'échec n'est plus masqué
    assert service.add({"plaque": " "})[0] is False


def test_validation_service_routes_actions():
    record = SimpleNamespace(plaque="AB123CD")
    detector = FakeMismatchDetector(pending=[record])
    service = ValidationService(detector)

    assert service.act("AB123CD", VALIDATE_PREDICTED) == "validé (modèle)"
    assert service.act("AB123CD", VALIDATE_DECLARED) == "validé (base)"
    assert service.act("AB123CD", REJECT) == "rejeté"
    assert service.act("AB123CD", CUSTOM, "Dacia") == "validé (Dacia)"
    assert service.act("AB123CD", CUSTOM, "") == "Saisissez la marque correcte"
    assert "non trouvée" in service.act("ZZ999ZZ", REJECT)
    assert ValidationService(None).act("AB123CD", REJECT) == "Détecteur d'incohérences non disponible"


def test_history_service_reads_logs():
    db = FakeDB()
    db.logs = [{"plaque": "AB123CD"}]
    assert HistoryService(db).recent() == [{"plaque": "AB123CD"}]


# ═══════════════════════════════════════════════════════════════════════════════
# MISE EN FORME
# ═══════════════════════════════════════════════════════════════════════════════

def test_formatters():
    from src.ui import formatters as fmt

    assert fmt.format_timestamp(datetime(2026, 10, 23, 9, 5, 1)) == "2026-10-23 09:05:01"
    assert fmt.format_timestamp("2026-10-23T09:05:01.123456") == "2026-10-23 09:05:01"

    service, _, _, _, _ = make_service({"AB123CD": RESIDENT_OTHER_BRAND})
    outcome = service.process(make_image())
    assert "MISMATCH DÉTECTÉ" in fmt.capture_report(outcome)
    assert "Peugeot" in fmt.brand_card(outcome.result.brand, outcome.mismatch, outcome.declared_brand)

    outcome.result.plates[0].ocr_confidence = None
    assert "N/A" in fmt.analysis_report(outcome.result)  # confiance absente : pas de chiffre inventé

    table = fmt.history_table([{"timestamp": datetime(2026, 10, 23), "plaque": "AB123CD", "mismatch": True}])
    assert list(table.columns) == fmt.HISTORY_COLUMNS and table.iloc[0]["Mismatch"] == "⚠️"


def test_ui_builds_with_services():
    """L'interface complète se construit à partir des services (sans modèles réels)."""
    pytest.importorskip("gradio")
    from src.services import Services
    from src.services.system import SystemService
    from src.ui.build import build_app
    from configs.settings import get_settings

    service, db, detector, _, engine = make_service({})
    services = Services(
        settings=get_settings(),
        capture=service,
        residents=ResidentService(db),
        validation=ValidationService(detector),
        history=HistoryService(db),
        system=SystemService(engine.engine, get_settings()),
    )
    assert build_app(services) is not None
