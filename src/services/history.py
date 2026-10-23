"""Historique des passages."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class HistoryService:
    def __init__(self, db):
        self.db = db

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            return self.db.get_logs(limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.error("Lecture de l'historique impossible : %s", exc)
            return []
