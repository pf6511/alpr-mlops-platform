"""
ALPR Engine - Ray Serve Deployment
Serving haute performance des modèles de détection et classification.

Endpoints:
- /detect_plate : Détection plaque + OCR
- /classify_brand : Classification marque véhicule
- /process : Pipeline complet (plaque + véhicule + marque)
- /health : Health check
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ray imports
try:
    import ray
    from ray import serve
    from ray.serve.handle import DeploymentHandle
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("⚠️ ray[serve] non installé")

# ML imports
try:
    import numpy as np
    import cv2
    from PIL import Image
    import io
    import base64
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Starlette pour HTTP
try:
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False

# Configuration
from configs.settings import get_settings

logger = logging.getLogger("ray.serve")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def decode_image(image_data: str) -> np.ndarray:
    """Décode une image base64 en numpy array BGR."""
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]
    
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image


def encode_image(image: np.ndarray, format: str = "jpeg") -> str:
    """Encode une image numpy en base64."""
    if format == "jpeg":
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        _, buffer = cv2.imencode('.png', image)
    
    return base64.b64encode(buffer).decode('utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENTS
# ═══════════════════════════════════════════════════════════════════════════════

if RAY_AVAILABLE:

    @serve.deployment(
        name="plate_detector",
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.25},
        max_concurrent_queries=10,
    )
    class PlateDetector:
        """Deployment pour détection de plaques + OCR."""
        
        def __init__(self):
            from ultralytics import YOLO
            from fast_plate_ocr import LicensePlateRecognizer
            
            model_path = Path("models") / "best_model_detection_plaque.pt"
            if not model_path.exists():
                model_path = Path("models") / "best.pt"
            if not model_path.exists():
                model_path = "yolov8n.pt"
            
            self.yolo = YOLO(str(model_path))
            self.ocr = LicensePlateRecognizer('global-plates-mobile-vit-v2-model')
            logger.info(f"✅ PlateDetector: chargé depuis {model_path}")
        
        async def __call__(self, request: Request) -> Dict:
            try:
                data = await request.json()
                image_b64 = data.get("image")
                conf = data.get("conf_threshold", 0.5)
                
                if not image_b64:
                    return JSONResponse({"error": "Image requise"}, status_code=400)
                
                image = decode_image(image_b64)
                results = self.yolo(image, conf=conf, verbose=False)
                
                plates = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        det_conf = float(box.conf[0])
                        
                        roi = image[y1:y2, x1:x2]
                        text, ocr_conf = "", 0.0
                        
                        if roi.size > 0:
                            try:
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                ocr_result = self.ocr.run(roi_gray, return_confidence=True)
                                if ocr_result and len(ocr_result) == 2:
                                    texts, confs = ocr_result
                                    if texts:
                                        text = texts[0].upper()
                                        ocr_conf = float(np.mean(confs[0]))
                            except Exception as e:
                                logger.warning(f"OCR error: {e}")
                        
                        plates.append({
                            "text": text,
                            "confidence": ocr_conf,
                            "detection_confidence": det_conf,
                            "bbox": [x1, y1, x2, y2]
                        })
                
                return JSONResponse({
                    "plates": plates,
                    "count": len(plates),
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"PlateDetector error: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)


    @serve.deployment(
        name="brand_classifier",
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.25},
        max_concurrent_queries=10,
    )
    class BrandClassifier:
        """Deployment pour classification de marque véhicule."""
        
        def __init__(self):
            from ultralytics import YOLO
            import torch
            from torchvision import transforms
            
            settings = get_settings()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.brand_classes = settings.models.brand_classes
            
            # YOLO véhicule
            vehicle_path = Path("models") / "yolov8s.pt"
            self.yolo_vehicle = YOLO(str(vehicle_path) if vehicle_path.exists() else "yolov8s.pt")
            
            # EfficientNet
            brand_path = Path("models") / "best_model_efficientNet_finetune.pth"
            self.brand_model = None
            
            if brand_path.exists():
                try:
                    import timm
                    state_dict = torch.load(brand_path, map_location=self.device)
                    
                    num_classes = len(self.brand_classes)
                    for key in state_dict.keys():
                        if 'classifier' in key and 'weight' in key:
                            num_classes = state_dict[key].shape[0]
                            break
                    
                    self.brand_model = timm.create_model(
                        'efficientnet_b4', pretrained=False, num_classes=num_classes
                    )
                    self.brand_model.load_state_dict(state_dict)
                    self.brand_model.to(self.device)
                    self.brand_model.eval()
                    
                    self.brand_transform = transforms.Compose([
                        transforms.Resize((380, 380)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    
                    logger.info(f"✅ BrandClassifier: EfficientNet chargé ({num_classes} classes)")
                    
                except Exception as e:
                    logger.error(f"Erreur EfficientNet: {e}")
            else:
                logger.warning(f"⚠️ {brand_path} non trouvé")
        
        async def __call__(self, request: Request) -> Dict:
            try:
                data = await request.json()
                image_b64 = data.get("image")
                conf = data.get("conf_threshold", 0.5)
                
                if not image_b64:
                    return JSONResponse({"error": "Image requise"}, status_code=400)
                
                image = decode_image(image_b64)
                
                # Détection véhicule
                vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
                results = self.yolo_vehicle(image, conf=conf, verbose=False)
                
                best_vehicle = None
                best_area = 0
                
                for r in results:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        if class_id in vehicle_classes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            area = (x2 - x1) * (y2 - y1)
                            if area > best_area:
                                best_area = area
                                best_vehicle = {
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": float(box.conf[0]),
                                    "class_name": vehicle_classes[class_id]
                                }
                
                if not best_vehicle:
                    return JSONResponse({"vehicle": None, "brand": None})
                
                # Classification marque
                brand_result = None
                if self.brand_model is not None:
                    x1, y1, x2, y2 = best_vehicle["bbox"]
                    vehicle_crop = image[y1:y2, x1:x2]
                    
                    if vehicle_crop.size > 0:
                        try:
                            import torch.nn.functional as F
                            from PIL import Image as PILImage
                            
                            rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
                            pil_img = PILImage.fromarray(rgb)
                            tensor = self.brand_transform(pil_img).unsqueeze(0).to(self.device)
                            
                            with torch.no_grad():
                                outputs = self.brand_model(tensor)
                                probs = F.softmax(outputs, dim=1)[0]
                            
                            top3_probs, top3_idx = torch.topk(probs, min(3, len(probs)))
                            top3 = [
                                {"brand": self.brand_classes[i] if i < len(self.brand_classes) else f"class_{i}",
                                 "confidence": float(p)}
                                for i, p in zip(top3_idx.tolist(), top3_probs.tolist())
                            ]
                            
                            brand_result = {
                                "name": top3[0]["brand"],
                                "confidence": top3[0]["confidence"],
                                "top3": top3
                            }
                        except Exception as e:
                            logger.error(f"Brand error: {e}")
                
                return JSONResponse({
                    "vehicle": best_vehicle,
                    "brand": brand_result,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"BrandClassifier error: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)


    @serve.deployment(
        name="alpr_pipeline",
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.5},
        max_concurrent_queries=5,
    )
    class ALPRPipelineServe:
        """Pipeline ALPR complet combinant PlateDetector + BrandClassifier."""
        
        def __init__(self, plate_detector: DeploymentHandle, brand_classifier: DeploymentHandle):
            self.plate_detector = plate_detector
            self.brand_classifier = brand_classifier
            logger.info("✅ ALPRPipelineServe: initialisé")
        
        async def __call__(self, request: Request) -> Dict:
            try:
                data = await request.json()
                image_b64 = data.get("image")
                declared_brand = data.get("declared_brand")
                return_images = data.get("return_images", False)
                
                if not image_b64:
                    return JSONResponse({"error": "Image requise"}, status_code=400)
                
                # Créer les requêtes pour les sous-services
                from starlette.requests import Request as StarletteRequest
                
                # Appels parallèles via handles
                plate_task = self.plate_detector.options(use_new_handle_api=True).remote(request)
                brand_task = self.brand_classifier.options(use_new_handle_api=True).remote(request)
                
                plate_result, brand_result = await asyncio.gather(plate_task, brand_task)
                
                plates = plate_result.get("plates", []) if isinstance(plate_result, dict) else []
                vehicle = brand_result.get("vehicle") if isinstance(brand_result, dict) else None
                brand = brand_result.get("brand") if isinstance(brand_result, dict) else None
                
                # Mismatch detection
                mismatch = False
                if declared_brand and brand:
                    mismatch = brand.get("name", "").lower() != declared_brand.lower()
                
                result = {
                    "plates": plates,
                    "vehicle": vehicle,
                    "brand": brand,
                    "mismatch": mismatch,
                    "declared_brand": declared_brand,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Image annotée
                if return_images:
                    try:
                        image = decode_image(image_b64)
                        
                        for plate in plates:
                            bbox = plate.get("bbox", [])
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = bbox
                                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                text = plate.get("text", "")
                                if text:
                                    cv2.putText(image, text, (x1, y1-10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        if vehicle and brand:
                            bbox = vehicle.get("bbox", [])
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = bbox
                                color = (0, 0, 255) if mismatch else (255, 165, 0)
                                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                                label = f"{brand.get('name', '')} ({brand.get('confidence', 0):.0%})"
                                cv2.putText(image, label, (x1, y2+20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        result["annotated_image"] = encode_image(image)
                    except Exception as e:
                        logger.warning(f"Annotation error: {e}")
                
                return JSONResponse(result)
                
            except Exception as e:
                logger.error(f"ALPRPipeline error: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)


    @serve.deployment(name="health", num_replicas=1)
    class HealthCheck:
        """Health check endpoint."""
        async def __call__(self, request: Request) -> Dict:
            return JSONResponse({
                "status": "healthy",
                "service": "alpr-engine",
                "timestamp": datetime.now().isoformat()
            })


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_app():
    """Construit l'application Ray Serve."""
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray Serve non disponible")
    
    plate_detector = PlateDetector.bind()
    brand_classifier = BrandClassifier.bind()
    pipeline = ALPRPipelineServe.bind(plate_detector, brand_classifier)
    
    return pipeline


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Démarre le serveur Ray Serve."""
    if not RAY_AVAILABLE:
        print("❌ Ray Serve non disponible")
        return
    
    settings = get_settings()
    ray_config = settings.ray
    
    if not ray.is_initialized():
        ray.init(
            address=ray_config.address if ray_config.address != "auto" else None,
            ignore_reinit_error=True
        )
    
    serve.start(http_options={"host": host, "port": port})
    app = build_app()
    serve.run(app, route_prefix="/")
    
    print(f"🚀 ALPR Ray Serve: http://{host}:{port}")
    print("   POST /          - Pipeline complet")
    print("   GET  /health    - Health check")


def stop_server():
    """Arrête le serveur."""
    if RAY_AVAILABLE:
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    start_server(args.host, args.port)
