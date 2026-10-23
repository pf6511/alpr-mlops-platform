"""Administration des modèles et informations système."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.inference import ALPREngine
from src.inference.loader import PROJECT_ROOT

MODELS_DIR = Path("models")


class SystemService:
    def __init__(self, engine: ALPREngine, settings):
        self.engine = engine
        self.settings = settings

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

    def info(self) -> Dict[str, Any]:
        return {
            "env": self.settings.env,
            "database": self.settings.database.mode,
            "storage": self.settings.s3.mode,
            "models": self.engine.status(),
        }
