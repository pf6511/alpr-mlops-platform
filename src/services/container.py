"""
Racine de composition : crée chaque composant une seule fois et les relie.

C'est le seul endroit qui sait « qui a besoin de quoi ». Les services reçoivent
leurs dépendances (injection) au lieu de les créer eux-mêmes : une seule
connexion à la base, un seul moteur d'inférence, partagés par tous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from configs.settings import Settings, get_settings
from src.data.database import DatabaseManager
from src.inference import ALPREngine
from src.utils.error_gallery import ErrorGallery

from .capture import ALPRService
from .history import HistoryService
from .residents import ResidentService
from .system import SystemService
from .validation import ValidationService

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    capture: ALPRService
    residents: ResidentService
    validation: ValidationService
    history: HistoryService
    system: SystemService


def _optional_storage():
    try:
        from src.data.storage import get_storage

        return get_storage()
    except Exception as exc:  # noqa: BLE001 - le stockage est facultatif
        logger.warning("Stockage indisponible : %s", exc)
        return None


def _optional_mismatch_detector(db, storage):
    try:
        from src.models.mismatch_detector import MismatchDetector

        return MismatchDetector(db=db, storage=storage)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Détecteur d'incohérences indisponible : %s", exc)
        return None


def build_services(settings: Optional[Settings] = None,
                   engine: Optional[ALPREngine] = None,
                   db=None) -> Services:
    """Assemble l'application. `engine` et `db` peuvent être fournis (tests)."""
    settings = settings or get_settings()
    db = db if db is not None else DatabaseManager()
    engine = engine or ALPREngine(settings=settings)
    storage = _optional_storage()
    detector = _optional_mismatch_detector(db, storage)

    return Services(
        settings=settings,
        capture=ALPRService(engine, db, mismatch_detector=detector, error_gallery=ErrorGallery()),
        residents=ResidentService(db),
        validation=ValidationService(detector),
        history=HistoryService(db),
        system=SystemService(engine, settings),
    )
