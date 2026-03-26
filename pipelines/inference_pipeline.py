"""
ALPR Engine - Inference Pipeline
Pipeline d'inférence pour la détection de plaques et classification de marques.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import time

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import get_settings

# Imports optionnels
try:
    import numpy as np
    import cv2
    from PIL import Image
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False


class InferencePipeline:
    """
    Pipeline d'inférence ALPR.
    
    Combine:
    - Détection de plaque (YOLOv8)
    - OCR (fast-plate-ocr)
    - Détection véhicule (YOLOv8s)
    - Classification marque (EfficientNet)
    - Détection de mismatch
    """
    
    def __init__(self, load_all: bool = True):
        """
        Initialise le pipeline.
        
        Args:
            load_all: Charger tous les modèles au démarrage
        """
        self.settings = get_settings()
        
        # Modèles
        self.yolo_plate = None
        self.yolo_vehicle = None
        self.ocr_model = None
        self.brand_model = None
        self.brand_transform = None
        
        # Device
        self.device = "cpu"
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass
        
        if load_all:
            self._load_models()
        
        print(f"✅ InferencePipeline initialisé (device: {self.device})")
    
    def _load_models(self):
        """Charge tous les modèles."""
        self._load_yolo_plate()
        self._load_yolo_vehicle()
        self._load_ocr()
        self._load_brand_classifier()
    
    def _load_yolo_plate(self):
        """Charge YOLO pour détection de plaques."""
        try:
            from ultralytics import YOLO
            
            model_path = Path(self.settings.models.yolo_plate_path)
            if not model_path.exists():
                model_path = Path("models/best_model_detection_plaque.pt")
            if not model_path.exists():
                model_path = "yolov8n.pt"
            
            self.yolo_plate = YOLO(str(model_path))
            print(f"   ✅ YOLO Plaque: {model_path}")
            
        except Exception as e:
            print(f"   ⚠️ YOLO Plaque: {e}")
    
    def _load_yolo_vehicle(self):
        """Charge YOLO pour détection de véhicules."""
        try:
            from ultralytics import YOLO
            
            model_path = Path(self.settings.models.yolo_vehicle_path)
            if not model_path.exists():
                model_path = Path("models/best_model_detection_vehicle.pt")
            if not model_path.exists():
                model_path = "yolov8s.pt"
            
            self.yolo_vehicle = YOLO(str(model_path))
            print(f"   ✅ YOLO Véhicule: {model_path}")
            
        except Exception as e:
            print(f"   ⚠️ YOLO Véhicule: {e}")
    
    def _load_ocr(self):
        """Charge le modèle OCR."""
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            
            self.ocr_model = LicensePlateRecognizer('global-plates-mobile-vit-v2-model')
            print("   ✅ OCR: global-plates-mobile-vit-v2-model")
            
        except Exception as e:
            print(f"   ⚠️ OCR: {e}")
    
    def _load_brand_classifier(self):
        """Charge EfficientNet pour classification de marques."""
        try:
            import torch
            import timm
            from torchvision import transforms
            
            model_path = Path(self.settings.models.efficientnet_path)
            
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Détecter num_classes
                num_classes = len(self.settings.models.brand_classes)
                for key in state_dict.keys():
                    if 'classifier' in key and 'weight' in key:
                        num_classes = state_dict[key].shape[0]
                        break
                
                self.brand_model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
                self.brand_model.load_state_dict(state_dict)
                self.brand_model.to(self.device)
                self.brand_model.eval()
                
                self.brand_transform = transforms.Compose([
                    transforms.Resize((380, 380)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                print(f"   ✅ EfficientNet: {model_path} ({num_classes} classes)")
            else:
                print(f"   ⚠️ EfficientNet: {model_path} non trouvé")
                
        except Exception as e:
            print(f"   ⚠️ EfficientNet: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INFERENCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_plates(self, image: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        """
        Détecte les plaques dans une image.
        
        Args:
            image: Image BGR numpy
            conf_threshold: Seuil de confiance
            
        Returns:
            Liste de {bbox, confidence, roi}
        """
        if self.yolo_plate is None:
            return []
        
        if conf_threshold is None:
            conf_threshold = self.settings.models.detection_threshold
        
        results = self.yolo_plate(image, conf=conf_threshold, verbose=False)
        
        plates = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                roi = image[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else None
                
                plates.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'roi': roi
                })
        
        return plates
    
    def read_plate(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Lit le texte d'une plaque.
        
        Args:
            roi: Image de la plaque (BGR)
            
        Returns:
            (texte, confiance)
        """
        if self.ocr_model is None or roi is None or roi.size == 0:
            return "", 0.0
        
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            result = self.ocr_model.run(gray, return_confidence=True)
            
            if result and len(result) == 2:
                texts, confs = result
                if texts:
                    text = texts[0].upper()
                    conf = float(np.mean(confs[0])) if confs else 0.0
                    return text, conf
        except Exception as e:
            print(f"OCR error: {e}")
        
        return "", 0.0
    
    def detect_vehicle(self, image: np.ndarray, conf_threshold: float = 0.5) -> Optional[Dict]:
        """
        Détecte le véhicule principal dans l'image.
        
        Args:
            image: Image BGR numpy
            conf_threshold: Seuil de confiance
            
        Returns:
            Dict avec bbox, class_name, confidence, crop
        """
        if self.yolo_vehicle is None:
            return None
        
        vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        
        results = self.yolo_vehicle(image, conf=conf_threshold, verbose=False)
        
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
                        crop = image[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else None
                        best_vehicle = {
                            'bbox': [x1, y1, x2, y2],
                            'class_name': vehicle_classes[class_id],
                            'confidence': float(box.conf[0]),
                            'crop': crop
                        }
        
        return best_vehicle
    
    def classify_brand(self, vehicle_crop: np.ndarray) -> Optional[Dict]:
        """
        Classifie la marque d'un véhicule.
        
        Args:
            vehicle_crop: Image du véhicule (BGR)
            
        Returns:
            Dict avec brand, confidence, top3
        """
        if self.brand_model is None or vehicle_crop is None or vehicle_crop.size == 0:
            return None
        
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image as PILImage
            
            # Préparer l'image
            rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            tensor = self.brand_transform(pil_img).unsqueeze(0).to(self.device)
            
            # Inférence
            with torch.no_grad():
                outputs = self.brand_model(tensor)
                probs = F.softmax(outputs, dim=1)[0]
            
            # Top 3
            top3_probs, top3_idx = torch.topk(probs, min(3, len(probs)))
            brand_classes = self.settings.models.brand_classes
            
            top3 = []
            for i, p in zip(top3_idx.tolist(), top3_probs.tolist()):
                brand = brand_classes[i] if i < len(brand_classes) else f"class_{i}"
                top3.append({'brand': brand, 'confidence': p})
            
            return {
                'brand': top3[0]['brand'],
                'confidence': top3[0]['confidence'],
                'top3': top3
            }
            
        except Exception as e:
            print(f"Brand classification error: {e}")
            return None
    
    def process(self, image: Union[str, np.ndarray], 
                conf_threshold: float = None,
                run_brand: bool = True,
                declared_brand: str = None) -> Dict:
        """
        Pipeline complet d'inférence.
        
        Args:
            image: Chemin ou image numpy
            conf_threshold: Seuil de confiance
            run_brand: Exécuter la classification de marque
            declared_brand: Marque déclarée (pour mismatch)
            
        Returns:
            Dict avec tous les résultats
        """
        start_time = time.time()
        
        # Charger l'image
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            return {'error': 'Image invalide'}
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'image_shape': image.shape,
            'plates': [],
            'vehicle': None,
            'brand': None,
            'mismatch': False,
            'declared_brand': declared_brand
        }
        
        # Détection plaques
        plates = self.detect_plates(image, conf_threshold)
        for plate in plates:
            text, ocr_conf = self.read_plate(plate['roi'])
            plate['text'] = text
            plate['ocr_confidence'] = ocr_conf
            del plate['roi']  # Ne pas retourner l'image
        result['plates'] = plates
        
        # Détection véhicule + marque
        if run_brand:
            vehicle = self.detect_vehicle(image)
            if vehicle:
                result['vehicle'] = {
                    'bbox': vehicle['bbox'],
                    'class_name': vehicle['class_name'],
                    'confidence': vehicle['confidence']
                }
                
                # Classification marque
                brand = self.classify_brand(vehicle['crop'])
                if brand:
                    result['brand'] = brand
                    
                    # Mismatch detection
                    if declared_brand:
                        pred = brand['brand'].lower()
                        decl = declared_brand.lower()
                        result['mismatch'] = pred != decl
        
        result['inference_time'] = time.time() - start_time
        
        return result
    
    def process_batch(self, images: List[Union[str, np.ndarray]], **kwargs) -> List[Dict]:
        """
        Traite un batch d'images.
        
        Args:
            images: Liste d'images
            **kwargs: Arguments pour process()
            
        Returns:
            Liste de résultats
        """
        return [self.process(img, **kwargs) for img in images]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="ALPR Inference Pipeline")
    parser.add_argument("image", help="Chemin de l'image")
    parser.add_argument("--no-brand", action="store_true", help="Désactiver classification marque")
    parser.add_argument("--declared-brand", help="Marque déclarée pour mismatch")
    parser.add_argument("--conf", type=float, default=0.5, help="Seuil de confiance")
    args = parser.parse_args()
    
    pipeline = InferencePipeline()
    result = pipeline.process(
        args.image,
        conf_threshold=args.conf,
        run_brand=not args.no_brand,
        declared_brand=args.declared_brand
    )
    
    print(json.dumps(result, indent=2, default=str))
