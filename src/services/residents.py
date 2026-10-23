"""Gestion des résidents (plaques autorisées et marque déclarée)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ResidentService:
    def __init__(self, db):
        self.db = db

    def add(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Ajoute un résident. Renvoie (succès, message) — un échec n'est plus masqué."""
        if not str(data.get("plaque", "")).strip():
            return False, "❌ La plaque est obligatoire."
        try:
            ok, message = self.db.add_resident(data)
        except Exception as exc:  # noqa: BLE001
            logger.error("Ajout du résident impossible : %s", exc)
            return False, f"❌ Erreur : {exc}"
        if ok:
            return True, f"✅ Résident {data.get('prenom', '')} {data.get('nom', '')} ajouté."
        return False, f"❌ {message}"

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            return self.db.get_all_residents()[:limit]
        except Exception as exc:  # noqa: BLE001
            logger.error("Lecture des résidents impossible : %s", exc)
            return []
