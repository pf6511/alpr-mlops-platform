"""
Registre des modèles de marque : le SEUL module du projet qui parle à MLflow.

Vocabulaire :
- version   : chaque modèle enregistré reçoit un numéro (1, 2, 3...)
- alias     : une étiquette mobile posée sur une version
              « champion »   = la version servie par l'application
              « challenger » = le dernier candidat, en attente de promotion

Un modèle est stocké comme un paquet (voir bundle.py) : des poids sans code,
ses classes et son manifeste de lignage. Le chargement n'exécute donc jamais de
code venu du registre, même si celui-ci est accessible publiquement.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs.settings import MLflowSettings, get_settings
from src.inference.loader import BUNDLE_WEIGHTS, BrandBundleRef

logger = logging.getLogger(__name__)

ARTIFACT_PATH = "brand_model"

# Tags posés sur les versions
VALIDATED_IMAGES_TAG = "validated_images"   # images validées utilisées pour l'entraîner
PROMOTION_TAG = "promotion"                 # décision de promotion (« promoted: … » / « rejected: … »)


@dataclass
class RegisteredBrandModel:
    version: str
    run_id: str
    source: str
    aliases: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


class BrandModelRegistry:
    def __init__(self, cfg: Optional[MLflowSettings] = None, client=None, mlflow_module=None):
        self.cfg = cfg or get_settings().mlflow
        self._client = client
        self._mlflow = mlflow_module

    # ═══════════════════════════════════════════════════════════════════════════
    # ACCÈS À MLFLOW (import paresseux : l'application démarre sans MLflow)
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def name(self) -> str:
        return self.cfg.brand_model_name

    @property
    def mlflow(self):
        if self._mlflow is None:
            import mlflow

            mlflow.set_tracking_uri(self.cfg.tracking_uri)
            self._mlflow = mlflow
        return self._mlflow

    @property
    def client(self):
        if self._client is None:
            from mlflow.tracking import MlflowClient

            self.mlflow  # fixe l'URI de suivi
            self._client = MlflowClient(tracking_uri=self.cfg.tracking_uri)
        return self._client

    # ═══════════════════════════════════════════════════════════════════════════
    # LECTURE
    # ═══════════════════════════════════════════════════════════════════════════

    def get(self, alias: str) -> Optional[RegisteredBrandModel]:
        """Version portant cet alias, ou None si l'alias n'existe pas."""
        try:
            mv = self.client.get_model_version_by_alias(self.name, alias)
        except Exception:  # noqa: BLE001 - alias absent ou modèle non enregistré
            return None
        return RegisteredBrandModel(
            version=str(mv.version), run_id=mv.run_id, source=mv.source,
            aliases=list(getattr(mv, "aliases", []) or []), tags=dict(getattr(mv, "tags", {}) or {}),
        )

    def versions(self) -> List[RegisteredBrandModel]:
        """Toutes les versions du modèle (liste vide si le modèle n'est pas encore enregistré)."""
        try:
            found = self.client.search_model_versions(f"name='{self.name}'")
        except Exception:  # noqa: BLE001 - modèle absent
            return []
        return [RegisteredBrandModel(version=str(mv.version), run_id=mv.run_id, source=mv.source,
                                     aliases=list(getattr(mv, "aliases", []) or []),
                                     tags=dict(getattr(mv, "tags", {}) or {})) for mv in found]

    def download(self, alias: Optional[str] = None) -> Optional[BrandBundleRef]:
        """Télécharge le paquet de la version portant `alias` (champion par défaut), avec cache local."""
        alias = alias or self.cfg.champion_alias
        model = self.get(alias)
        if model is None:
            logger.warning("Aucune version « %s » pour %s dans le registre.", alias, self.name)
            return None

        target = Path(self.cfg.cache_dir) / self.name / f"v{model.version}"
        label = f"registry:{self.name}@{alias} (v{model.version})"
        if (target / BUNDLE_WEIGHTS).exists():
            return BrandBundleRef(directory=target, label=label)

        tmp = target.with_name(target.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        local = Path(self.mlflow.artifacts.download_artifacts(artifact_uri=model.source, dst_path=str(tmp)))
        bundle_dir = local if (local / BUNDLE_WEIGHTS).exists() else next(local.rglob(BUNDLE_WEIGHTS)).parent
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(bundle_dir), str(target))
        shutil.rmtree(tmp, ignore_errors=True)
        logger.info("Modèle téléchargé depuis le registre : %s", label)
        return BrandBundleRef(directory=target, label=label)

    # ═══════════════════════════════════════════════════════════════════════════
    # ÉCRITURE
    # ═══════════════════════════════════════════════════════════════════════════

    def register(self, bundle_dir: Path, *, run_name: str, params: Dict[str, Any],
                 metrics: Dict[str, float], tags: Dict[str, str],
                 history: Optional[List[Dict[str, float]]] = None,
                 alias: Optional[str] = None) -> RegisteredBrandModel:
        """Journalise une exécution (paramètres, métriques, paquet) et enregistre une nouvelle version."""
        mlflow = self.mlflow
        mlflow.set_experiment(self.cfg.experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            for step, epoch_metrics in enumerate(history or []):
                mlflow.log_metrics(epoch_metrics, step=step)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(tags)
            mlflow.log_artifacts(str(bundle_dir), artifact_path=ARTIFACT_PATH)
            run_id = run.info.run_id
            source = f"{run.info.artifact_uri.rstrip('/')}/{ARTIFACT_PATH}"

        self._ensure_registered_model()
        mv = self.client.create_model_version(name=self.name, source=source, run_id=run_id, tags=tags)
        self._wait_until_ready(str(mv.version))
        if alias:
            self.set_alias(alias, str(mv.version))
        logger.info("Version %s de %s enregistrée%s", mv.version, self.name, f" ({alias})" if alias else "")
        return RegisteredBrandModel(version=str(mv.version), run_id=run_id, source=source,
                                    aliases=[alias] if alias else [], tags=tags)

    def set_alias(self, alias: str, version: str) -> None:
        self.client.set_registered_model_alias(self.name, alias, str(version))

    def tag_version(self, version: str, key: str, value: str) -> None:
        self.client.set_model_version_tag(self.name, str(version), key, value)

    def _ensure_registered_model(self) -> None:
        try:
            self.client.get_registered_model(self.name)
        except Exception:  # noqa: BLE001 - n'existe pas encore
            self.client.create_registered_model(
                self.name, description="Classificateur de marque ALPR (EfficientNet B0, torchvision)"
            )

    def _wait_until_ready(self, version: str, timeout_s: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = getattr(self.client.get_model_version(self.name, version), "status", "READY")
            if status in ("READY", None) or str(status).endswith("READY"):
                return
            if "FAILED" in str(status):
                raise RuntimeError(f"Enregistrement de la version {version} en échec")
            time.sleep(1)
        raise TimeoutError(f"Version {version} toujours en cours d'enregistrement après {timeout_s} s")
