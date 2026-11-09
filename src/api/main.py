"""
ALPR - API REST du traitement d'un passage de véhicule (front office des équipements).

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Equivalent de app.py pour les machines : 
- ne contient pas de règle
- assemble les mêmes services que l'interface (build_services)
- expose

    GET  /health      état du service (ouvert, pour la supervision)
    POST /v1/access   image d'un véhicule → décision d'accès motivée (clé d'API requise)

La gestion des résidents la validation des incohérences  l'historique etc (ce qui apparaît
dans l'app) reste dans le "back office Gradio" : la clé d'une borne ne donne droit qu'à une seule action.

Sécurité :
    - clé d'API dans l'en-tête X-API-Key (variable ALPR_API_KEY, dans docker/.env)
      sans clé configurée, l'API refuse de démarrer ;
    - taille d'image limitée (ALPR_API_MAX_IMAGE_MB, 5 Mo par défaut), JPEG ou PNG
    - erreurs internes : message générique, sans détail technique
    - minimisation (RGPD) : ni nom du résident, ni image dans la réponse
    - le texte de la plaque n'est pas écrit dans les logs de l'API.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from src.inference.images import decode_image_bytes
from src.services.capture import REASON_NO_PLATE

logger = logging.getLogger("alpr.api")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}


# ------------------------------------------------------------------------------
# CONTRAT : schémas des réponses
# ------------------------------------------------------------------------------

class Health(BaseModel):
    status: str
    models_ready: bool
    brand_model: str


class Access(BaseModel):
    authorized: bool
    reason: str               # AUTHORIZED, UNKNOWN_PLATE, ACCESS_REVOKED, NO_PLATE


class Brand(BaseModel):
    predicted: str
    confidence: float


class AccessResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())   # autorise le champ « model_version »

    request_id: str
    plate: Optional[str]      # None si aucune plaque lisible
    access: Access
    brand: Optional[Brand]
    mismatch: bool            # marque prédite <> marque déclarée : image envoyée en validation
    model_version: str
    processing_ms: int


class ApiError(Exception):
    """Erreur renvoyée à l'appelant, au format commun {error, message, request_id}."""

    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


# ------------------------------------------------------------------------------
# APPLICATION
# -----------------------------------------------------------------------------+-

def create_app(services=None, api_key: Optional[str] = None,
               max_image_mb: Optional[float] = None) -> FastAPI:
    """
    Construit l'API. Les paramètres permettent de faciliter les tests
    - en production, tout vient de l'environnement et les services sont assemblés au démarrage.
    """
    api_key = api_key if api_key is not None else os.environ.get("ALPR_API_KEY", "")
    max_bytes = int((max_image_mb or float(os.environ.get("ALPR_API_MAX_IMAGE_MB", 5))) * 1024 * 1024)
    inference_lock = threading.Lock()   # les modèles ne sont pas garantis thread-safe

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Pas de clé, pas d'API : on refuse de démarrer plutôt que d'être ouvert
        if not api_key:
            raise RuntimeError("ALPR_API_KEY non définie : l'API refuse de démarrer")
        # 2. Mêmes services que l'interface Gradio (même moteur, même champion, même base)
        if services is not None:
            app.state.services = services
        else:
            from src.services import build_services

            app.state.services = build_services()
            settings = app.state.services.settings
            app.state.services.system.start_champion_watch(settings.continuous.champion_refresh_minutes)
        yield

    app = FastAPI(
        title="ALPR - API de contrôle d'accès de véhicule",
        version="1.0.0",
        description="Traitement d'un passage : image du véhicule -> décision d'accès motivée.",
        lifespan=lifespan,
    )

    # ------- Erreurs : format commun, jamais de détail interne --------------------------
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content={
            "error": exc.code, "message": exc.message, "request_id": request.state.request_id,
        })

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("Erreur interne (request_id=%s)", request_id)
        return JSONResponse(status_code=500, content={
            "error": "INTERNAL_ERROR", "message": "Erreur interne", "request_id": request_id,
        })

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    # -- Sécurité : clé d'API ------------------------------------------------
    def require_api_key(x_api_key: str = Header(default="")) -> None:
        # compare_digest : comparaison en temps constant (pas d'indice par le temps de réponse)
        if not secrets.compare_digest(x_api_key.encode(), api_key.encode()):
            raise ApiError(401, "UNAUTHORIZED", "Clé d'API absente ou invalide")

    def brand_model_label(svc) -> str:
        return svc.system.engine.models.sources.get("brand_classifier", "inconnu")

    # -- End-points ----------------------------------------------------------
    @app.get("/health", response_model=Health)
    def health(request: Request):
        svc = request.app.state.services
        engine = svc.system.engine
        return Health(
            status="ok",
            models_ready=engine.models.plate_detector is not None and engine.models.plate_reader is not None,
            brand_model=brand_model_label(svc),
        )

    @app.post("/v1/access", response_model=AccessResponse, dependencies=[Depends(require_api_key)])
    def access(request: Request,
               image: UploadFile = File(..., description="Photo du véhicule (JPEG ou PNG)"),
               gate_id: str = Form("", description="Identifiant de la borne, ex. entree-nord")):
        start = time.perf_counter()
        svc = request.app.state.services

        # 1. Contrôles d'entrée (avant tout calcul)
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            formats = ", ".join(sorted(t.split("/", 1)[1] for t in ALLOWED_IMAGE_TYPES))
            raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", f"Formats acceptés : {formats}")
        data = image.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ApiError(413, "IMAGE_TOO_LARGE", f"Image limitée à {max_bytes // (1024 * 1024)} Mo")
        try:
            image_bgr = decode_image_bytes(data)
        except ValueError:
            raise ApiError(400, "INVALID_IMAGE", "Image illisible") from None
        if svc.system.engine.models.plate_detector is None:
            raise ApiError(503, "MODELS_NOT_READY", "Modèles en cours de chargement")

        # 2. Traitement du passage : LE MÊME service que l'onglet Capture
        #    (analyse, résident, décision, cohérence de la marque, journal, image si incohérence)
        # les modèles ne sont pas garantis thread-safe
        with inference_lock:
            outcome = svc.capture.process(image_bgr)

        # 3. Réponse minimale (contrat)
        if outcome.access is None:      # aucune plaque lisible : pas de décision possible
            decision = Access(authorized=False, reason=REASON_NO_PLATE)
        else:
            decision = Access(authorized=outcome.access.authorized, reason=outcome.access.reason)
        brand = outcome.result.brand
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # Log technique, sans la plaque (minimisation)
        logger.info("request_id=%s gate=%s reason=%s mismatch=%s %d ms",
                    request.state.request_id, gate_id or "-", decision.reason, outcome.mismatch, elapsed_ms)

        return AccessResponse(
            request_id=request.state.request_id,
            plate=outcome.plate_text or None,
            access=decision,
            brand=Brand(predicted=brand.brand, confidence=round(brand.confidence, 4)) if brand else None,
            mismatch=outcome.mismatch,
            model_version=brand_model_label(svc),
            processing_ms=elapsed_ms,
        )

    return app


app = create_app()
