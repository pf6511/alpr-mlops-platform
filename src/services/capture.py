"""
Cas d'usage principal : traiter le passage d'un véhicule.

    photo → analyse → résident ? → accès → marque cohérente ? → journal

Toute la logique métier d'un passage est ici, et seulement ici. Le service ne
connaît pas Gradio : il reçoit une image et renvoie un `CaptureOutcome`, que
n'importe quelle interface (Gradio, API REST...) peut afficher.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.inference import ALPREngine, InferenceResult, is_brand_mismatch

logger = logging.getLogger(__name__)


@dataclass
class AccessDecision:
    """Décision d'accès pour une plaque lue."""

    authorized: bool
    message: str


@dataclass
class CaptureOutcome:
    """Tout ce qu'il faut savoir d'un passage, pour l'afficher ou l'exposer."""

    image_bgr: np.ndarray
    result: InferenceResult
    plate_text: str = ""                     # "" si aucune plaque lisible
    resident: Optional[Dict[str, Any]] = None
    declared_brand: Optional[str] = None
    access: Optional[AccessDecision] = None  # None si aucune plaque lisible
    mismatch: bool = False                   # marque prédite ≠ marque déclarée
    mismatch_image_key: Optional[str] = None # image du véhicule conservée pour validation
    failure_recorded: bool = False           # ajouté à la galerie d'erreurs


def decide_access(plate_text: str, resident: Optional[Dict[str, Any]]) -> AccessDecision:
    """Règle d'accès : la plaque appartient à un résident dont l'accès est « oui »."""
    authorized = bool(resident) and str(resident.get("acces", "")).strip().lower() == "oui"
    if authorized:
        return AccessDecision(True, f"✅ ACCESS GRANTED: {plate_text}")
    return AccessDecision(False, f"⛔ ACCESS DENIED: {plate_text}")


class ALPRService:
    """Traitement complet d'un passage de véhicule."""

    def __init__(self, engine: ALPREngine, db, mismatch_detector=None, error_gallery=None):
        self.engine = engine
        self.db = db
        self.mismatch_detector = mismatch_detector
        self.error_gallery = error_gallery

    def process(self, image_bgr: np.ndarray, conf_threshold: Optional[float] = None,
                run_brand: bool = True) -> CaptureOutcome:
        # 1. Analyse de l'image — une seule fois
        result = self.engine.analyze(image_bgr, conf_threshold=conf_threshold, run_branch2=run_brand)
        outcome = CaptureOutcome(image_bgr=image_bgr, result=result)

        plate = result.best_plate
        if plate is None:
            # Aucune plaque lisible : pas de décision d'accès, cas conservé pour analyse
            outcome.failure_recorded = self._record_failure(image_bgr, result)
            return outcome
        outcome.plate_text = plate.text

        # 2. Résident : une seule requête donne l'accès ET la marque déclarée
        outcome.resident = self._find_resident(plate.text)
        if outcome.resident:
            outcome.declared_brand = outcome.resident.get("marque_declaree") or None

        # 3. Décision d'accès, toujours à partir de la base (pas de cache périmé)
        outcome.access = decide_access(plate.text, outcome.resident)

        # 4. Cohérence de la marque
        predicted = result.brand.brand if result.brand else None
        outcome.mismatch = is_brand_mismatch(predicted, outcome.declared_brand)

        # 5. Incohérence : image du véhicule conservée pour la validation humaine
        if outcome.mismatch:
            outcome.mismatch_image_key = self._record_mismatch(outcome)

        # 6. Journal : UNE entrée par passage, reliée à l'image en cas d'incohérence
        self._log_passage(outcome)

        return outcome

    # ═══════════════════════════════════════════════════════════════════════════
    # ÉTAPES
    # ═══════════════════════════════════════════════════════════════════════════

    def _find_resident(self, plate_text: str) -> Optional[Dict[str, Any]]:
        try:
            return self.db.get_resident_by_plate(plate_text)
        except Exception as exc:  # noqa: BLE001 - la base indisponible ne doit pas bloquer l'analyse
            logger.error("Recherche du résident impossible : %s", exc)
            return None

    def _log_passage(self, outcome: CaptureOutcome) -> None:
        brand = outcome.result.brand
        self.db.add_log(
            plaque=outcome.plate_text,
            authorized=outcome.access.authorized,
            normalized=outcome.plate_text,
            marque_predite=brand.brand if brand else None,
            marque_confiance=brand.confidence if brand else None,
            mismatch=outcome.mismatch,
            image_key=outcome.mismatch_image_key,
        )

    def _record_mismatch(self, outcome: CaptureOutcome) -> Optional[str]:
        """Sauvegarde l'image du véhicule ; renvoie sa clé de stockage (ou None)."""
        if self.mismatch_detector is None:
            return None
        vehicle = outcome.result.vehicle
        crop_rgb = None
        if vehicle is not None and vehicle.crop is not None:
            crop_rgb = cv2.cvtColor(vehicle.crop, cv2.COLOR_BGR2RGB)  # le stockage attend du RGB
        record = self.mismatch_detector.detect_and_record(
            plaque=outcome.plate_text,
            marque_predite=outcome.result.brand.brand,
            marque_declaree=outcome.declared_brand,
            confiance=outcome.result.brand.confidence,
            vehicle_crop=crop_rgb,
            log_to_db=False,  # le passage est journalisé par ce service
        )
        return record.image_path if record is not None else None

    def _record_failure(self, image_bgr: np.ndarray, result: InferenceResult) -> bool:
        """Ajoute l'image à la galerie d'erreurs (fichier temporaire propre à l'appel)."""
        if self.error_gallery is None:
            return False
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            cv2.imwrite(path, image_bgr)
            detection_conf = max((p.detection_confidence for p in result.plates), default=0.0)
            self.error_gallery.add_example(
                image_path=path,
                issue="No Plate Detected" if not result.plates else "Plate Not Readable",
                expected="Unknown",
                predicted="None",
                detection_conf=detection_conf,
                ocr_conf=0.0,
                analysis=f"Échec de lecture de plaque ({datetime.now():%Y-%m-%d %H:%M}).",
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Galerie d'erreurs non mise à jour : %s", exc)
            return False
        finally:
            os.remove(path)
