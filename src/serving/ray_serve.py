"""
ALPR Engine - Adaptateur Ray Serve du cœur d'inférence.

Endpoints :
- POST /         : analyse complète (plaques + véhicule + marque)
- POST /process  : alias de /
- GET  /health   : état du service et des modèles

Corps de la requête POST (JSON) :
    {"image": "<base64>", "conf_threshold": 0.25, "run_brand": true,
     "declared_brand": "renault", "return_images": false}

La réponse a le même format que la CLI (`InferenceResult.to_dict()`), complété
de `mismatch`, `declared_brand` et, sur demande, `annotated_image` (JPEG base64).

Variables d'environnement : RAY_REPLICAS (défaut 1), RAY_NUM_GPUS (défaut 0,
fraction de GPU par réplique — à laisser à 0 sur une machine sans GPU).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import get_settings  # noqa: E402
from src.inference import ALPREngine, is_brand_mismatch  # noqa: E402
from src.inference.rendering import annotate  # noqa: E402

try:
    import ray
    from ray import serve
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger("ray.serve")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════════════

def decode_image(image_data: str) -> np.ndarray:
    """Image base64 (éventuellement préfixée « data:...;base64, ») → numpy BGR."""
    if "base64," in image_data:
        image_data = image_data.split("base64,", 1)[1]
    buffer = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image illisible")
    return image


def encode_image(image_bgr: np.ndarray) -> str:
    """Image numpy BGR → JPEG base64."""
    ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Encodage JPEG impossible")
    return base64.b64encode(buffer).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIQUE DE L'ENDPOINT (indépendante de Ray, donc testable)
# ═══════════════════════════════════════════════════════════════════════════════

def handle_process(engine: ALPREngine, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Traite le corps JSON d'une requête d'analyse. Lève ValueError si invalide."""
    image_b64 = payload.get("image")
    if not image_b64:
        raise ValueError("Champ 'image' (base64) requis")

    image = decode_image(image_b64)
    result = engine.analyze(
        image,
        conf_threshold=payload.get("conf_threshold"),
        run_branch2=bool(payload.get("run_brand", True)),
    )

    declared_brand = payload.get("declared_brand")
    mismatch = is_brand_mismatch(result.brand.brand if result.brand else None, declared_brand)

    response = result.to_dict()
    response["declared_brand"] = declared_brand
    response["mismatch"] = mismatch

    if payload.get("return_images"):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated = cv2.cvtColor(annotate(rgb, result, mismatch=mismatch), cv2.COLOR_RGB2BGR)
        response["annotated_image"] = encode_image(annotated)

    return response


def health_payload(engine: ALPREngine) -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "alpr-engine",
        "models": engine.status(),
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DÉPLOIEMENT
# ═══════════════════════════════════════════════════════════════════════════════

if RAY_AVAILABLE:
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @serve.deployment(
        name="alpr_engine",
        num_replicas=int(os.environ.get("RAY_REPLICAS", 1)),
        ray_actor_options={"num_gpus": float(os.environ.get("RAY_NUM_GPUS", 0))},
    )
    class ALPRDeployment:
        """Une réplique = un moteur d'inférence (modèles chargés une fois)."""

        def __init__(self):
            self.engine = ALPREngine()
            # Les modèles ne sont pas garantis thread-safe : une inférence à la fois par réplique.
            self._lock = threading.Lock()
            logger.info("ALPRDeployment initialisé : %s", self.engine.status())

        def _process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            with self._lock:
                return handle_process(self.engine, payload)

        async def __call__(self, request: Request):
            path = request.url.path.rstrip("/")
            if request.method == "GET" and path.endswith("/health"):
                return JSONResponse(health_payload(self.engine))
            if request.method != "POST" or path not in ("", "/process"):
                return JSONResponse({"error": "Route inconnue"}, status_code=404)
            try:
                payload = await request.json()
                # L'inférence (bloquante) tourne hors de la boucle asynchrone
                return JSONResponse(await asyncio.to_thread(self._process, payload))
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            except Exception as exc:  # noqa: BLE001
                logger.error("Erreur d'inférence : %s", exc)
                return JSONResponse({"error": str(exc)}, status_code=500)


def build_app():
    """Construit l'application Ray Serve."""
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray Serve non disponible : pip install 'ray[serve]'")
    return ALPRDeployment.bind()


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Démarre le serveur Ray Serve."""
    if not RAY_AVAILABLE:
        print("❌ Ray Serve non disponible")
        return

    ray_config = get_settings().ray
    if not ray.is_initialized():
        ray.init(
            address=ray_config.address if ray_config.address != "auto" else None,
            ignore_reinit_error=True,
        )
    serve.start(http_options={"host": host, "port": port})
    serve.run(build_app(), route_prefix="/")

    print(f"🚀 ALPR Ray Serve : http://{host}:{port}")
    print("   POST /          - Analyse complète")
    print("   GET  /health    - État du service")


def stop_server():
    """Arrête le serveur."""
    if RAY_AVAILABLE:
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    start_server(args.host, args.port)
