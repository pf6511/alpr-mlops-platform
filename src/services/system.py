"""Administration des modèles et informations système."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.inference import ALPREngine
from src.inference.loader import PROJECT_ROOT

MODELS_DIR = Path("models")

logger = logging.getLogger(__name__)


class SystemService:
    def __init__(self, engine: ALPREngine, settings, registry=None):
        self.engine = engine
        self.settings = settings
        self.registry = registry      # BrandModelRegistry, ou None si le registre n'est pas utilisé

    def available_plate_models(self) -> List[str]:
        """Fichiers de modèles (.pt, .pth) présents dans models/."""
        for directory in (MODELS_DIR, PROJECT_ROOT / MODELS_DIR):
            if directory.exists():
                files = sorted(f.name for f in directory.iterdir() if f.suffix in (".pt", ".pth"))
                if files:
                    return files
        return []

    def current_plate_model(self) -> str:
        source = self.engine.models.sources.get("plate_detector", "")
        return Path(source).name if source else "N/A"

    def reload_plate_model(self, model_name: str) -> Tuple[bool, str]:
        if not model_name:
            return False, "❌ Aucun modèle sélectionné"
        return self.engine.reload_plate_detector(model_name)

    def reload_brand_champion(self) -> Tuple[bool, str]:
        """Télécharge le champion actuel du registre et l'installe sans redémarrage."""
        if self.registry is None:
            return False, "ℹ️ Registre désactivé (MODEL_BRAND_SOURCE=local)"
        try:
            bundle = self.registry.download()
        except Exception as exc:  # noqa: BLE001 - registre injoignable
            return False, f"❌ Registre injoignable : {exc}"
        if bundle is None:
            return False, "❌ Aucun champion dans le registre"
        return self.engine.replace_brand_classifier(bundle)

    def refresh_champion_if_changed(self) -> Tuple[bool, str]:
        """Charge le champion du registre seulement si sa version a changé (appel périodique)."""
        if self.registry is None:
            return False, "registre désactivé"
        champion = self.registry.get(self.settings.mlflow.champion_alias)
        if champion is None:
            return False, "aucun champion dans le registre"
        current = self.engine.models.sources.get("brand_classifier", "")
        if current.startswith("registry:") and current.endswith(f"(v{champion.version})"):
            return False, f"champion inchangé (v{champion.version})"
        logger.info("Nouveau champion v%s dans le registre : rechargement", champion.version)
        return self.reload_brand_champion()

    def start_champion_watch(self, interval_minutes: float) -> Optional[threading.Thread]:
        """
        Vérifie le champion toutes les `interval_minutes` en tâche de fond : une promotion
        par le réentraînement automatique est prise en compte sans redémarrer l'application.
        """
        if self.registry is None or interval_minutes <= 0:
            return None

        def watch():
            stop = threading.Event()
            while not stop.wait(interval_minutes * 60):
                try:
                    changed, message = self.refresh_champion_if_changed()
                    if changed:
                        logger.info(message)
                except Exception as exc:  # noqa: BLE001 - registre injoignable : on garde le modèle actuel
                    logger.warning("Vérification du champion impossible : %s", exc)

        thread = threading.Thread(target=watch, name="champion-watch", daemon=True)
        thread.start()
        logger.info("Surveillance du champion : toutes les %s min", interval_minutes)
        return thread

    def info(self) -> Dict[str, Any]:
        return {
            "env": self.settings.env,
            "database": self.settings.database.mode,
            "storage": self.settings.s3.mode,
            "models": self.engine.status(),
        }
