"""
Validation humaine des incohérences de marque.

Un opérateur confirme la bonne marque ; les images validées alimentent le jeu
de données de réentraînement (boucle d'amélioration continue).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Actions possibles sur une incohérence
VALIDATE_PREDICTED = "validate_predicted"   # le modèle avait raison, la base se trompe
VALIDATE_DECLARED = "validate_declared"     # la base avait raison, le modèle se trompe
REJECT = "reject"                           # image inexploitable
CUSTOM = "custom"                           # ni l'un ni l'autre : marque saisie


class ValidationService:
    def __init__(self, mismatch_detector=None, retraining_progress: Optional[Callable[[], Dict]] = None):
        self.detector = mismatch_detector
        # Progression vers le prochain réentraînement (même règle que src.mlops.continuous)
        self.retraining_progress = retraining_progress

    @property
    def available(self) -> bool:
        return self.detector is not None

    def pending(self, limit: int = 20) -> List[Any]:
        return self.detector.get_pending(limit=limit) if self.detector else []

    def stats(self) -> Dict[str, Any]:
        stats = self.detector.get_stats() if self.detector else {}
        if self.retraining_progress is not None:
            try:
                stats["retraining"] = self.retraining_progress()
            except Exception as exc:  # noqa: BLE001 - registre ou stockage injoignable
                logger.warning("Progression du réentraînement indisponible : %s", exc)
        return stats

    def act(self, plaque: str, action: str, correct_brand: str = "") -> str:
        """Applique une décision de l'opérateur à l'incohérence de cette plaque."""
        if not self.detector:
            return "Détecteur d'incohérences non disponible"
        if not plaque:
            return "Sélectionnez une plaque dans la liste"

        record = next((r for r in self.detector.get_pending(limit=100) if r.plaque == plaque), None)
        if record is None:
            return f"Plaque {plaque} non trouvée dans la file"

        if action == VALIDATE_PREDICTED:
            _, message = self.detector.validate_as_predicted(record)
        elif action == VALIDATE_DECLARED:
            _, message = self.detector.validate_as_declared(record)
        elif action == REJECT:
            _, message = self.detector.reject(record, "Rejet manuel")
        else:
            if not correct_brand.strip():
                return "Saisissez la marque correcte"
            _, message = self.detector.validate(record, correct_brand.strip())
        return message
