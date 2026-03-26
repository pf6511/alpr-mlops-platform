"""
ALPR Engine - Pipeline de traitement (LEAD)
Branche 1: Détection plaque + OCR
Branche 2: Détection véhicule + Classification marque (EfficientNet)
"""

import os
import gc
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import get_settings
from pathlib import Path

MODELS_DIR = Path("models")

# Custom YOLO support (from original project)
try:
    from .custom_yolo import SimpleYOLO, load_custom_model
    CUSTOM_YOLO_AVAILABLE = True
except ImportError:
    CUSTOM_YOLO_AVAILABLE = False

# ML imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics non installé")

try:
    from fast_plate_ocr import LicensePlateRecognizer
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ fast-plate-ocr non installé")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ torch non installé")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️ timm non installé - EfficientNet via timm non disponible")


# ═══════════════════════════════════════════════════════════════════════════════
# NOMS DES MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PLATE = "best_model_detection_plaque.pt"
MODEL_VEHICLE = "yolov8s.pt"  # Pré-entraîné COCO, détecte voitures/motos/bus/camions
MODEL_BRAND = "best_model_efficientNet_finetune.pth"

# Modèles custom (SimpleYOLO from scratch)
CUSTOM_MODELS = ["modelemaison.pt", "YOLO_From_Scratch_LicensePlatev2.pt"]


class ALPRPipeline:
    """
    Pipeline ALPR complet avec 4 modèles.
    
    Branche 1 (plaque):
        - YOLOv8 → Détection plaque
        - fast-plate-ocr → Lecture caractères
    
    Branche 2 (véhicule):
        - YOLOv8s → Détection véhicule (crop)
        - EfficientNet → Classification marque (22 marques EU)
    """
    
    def __init__(self, model_path: str = None, load_branch2: bool = True):
        """
        Initialise le pipeline.
        
        Args:
            model_path: Chemin vers modèle YOLO plaque (override)
            load_branch2: Charger les modèles branche 2 (véhicule + marque)
        """
        self.settings = get_settings()
        self.model_config = self.settings.models
        
        # État des modèles
        self.yolo_plate = None
        self.yolo_vehicle = None
        self.ocr_model = None
        self.brand_model = None
        self.brand_transform = None
        
        # Flags
        self.is_custom_model = False
        self.model_path = model_path
        
        # Device
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        print(f"🖥️ Device: {self.device}")
        
        # Charger les modèles
        self._load_models(load_branch2)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _find_model(self, model_name: str) -> Optional[str]:
        """Cherche un modèle dans différents emplacements."""
        search_paths = [
            MODELS_DIR / model_name,
            Path("models") / model_name,
            Path(model_name),
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _load_models(self, load_branch2: bool = True):
        """Charge tous les modèles."""
        print("📦 Chargement des modèles...")
        
        # Branche 1: YOLO Plaque
        self._load_yolo_plate()
        
        # Branche 1: OCR
        self._load_ocr()
        
        if load_branch2:
            # Branche 2: YOLO Véhicule
            self._load_yolo_vehicle()
            
            # Branche 2: EfficientNet
            self._load_brand_classifier()
        
        print("✅ Modèles chargés!")
    
    def _load_yolo_plate(self):
        """Charge le modèle YOLO pour détection de plaques."""
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO non disponible")
            return
        
        # Chercher le modèle
        if self.model_path:
            model_path = self.model_path
        else:
            model_path = self._find_model(MODEL_PLATE)
        
        if model_path is None:
            # Fallback: chercher best.pt ou yolov8n.pt
            model_path = self._find_model("best.pt") or "yolov8n.pt"
            print(f"⚠️ {MODEL_PLATE} non trouvé, fallback: {model_path}")
        
        self.model_path = model_path
        model_name = os.path.basename(model_path)
        
        try:
            # Vérifier si c'est un modèle custom (SimpleYOLO)
            if model_name in CUSTOM_MODELS and CUSTOM_YOLO_AVAILABLE:
                print(f"🏠 Chargement modèle custom: {model_name}")
                self.yolo_plate = load_custom_model(model_path, device=self.device)
                self.is_custom_model = True
            else:
                self.yolo_plate = YOLO(model_path)
                self.is_custom_model = False
            
            print(f"✅ YOLO Plaque: {model_path}")
            
        except Exception as e:
            print(f"❌ Erreur chargement YOLO Plaque: {e}")
    
    def _load_ocr(self):
        """Charge le modèle OCR pour lecture de plaques."""
        if not OCR_AVAILABLE:
            print("⚠️ OCR non disponible")
            return
        
        try:
            self.ocr_model = LicensePlateRecognizer('global-plates-mobile-vit-v2-model')
            print("✅ OCR: fast-plate-ocr (global-plates-mobile-vit-v2-model)")
        except Exception as e:
            print(f"⚠️ Erreur chargement OCR: {e}")
            self.ocr_model = None

    def _load_brand_classifier(self):
        """Charge le modèle EfficientNet pour classification marque."""
        if not TORCH_AVAILABLE:
            print("⚠️ EfficientNet non disponible (torch manquant)")
            return
        
        model_path = self._find_model(MODEL_BRAND)
        
        if model_path is None:
            print(f"⚠️ {MODEL_BRAND} non trouvé - Classification marque désactivée")
            return
        
        try:
            # Charger le checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extraire le state_dict (format checkpoint vs direct)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Récupérer les classes depuis le checkpoint
                if 'label_encoder_classes' in checkpoint:
                    self._brand_classes = checkpoint['label_encoder_classes']
                # Priorité au num_classes du checkpoint
                num_classes = checkpoint.get('num_classes', 22)
                print(f"📦 Checkpoint EfficientNet: epoch={checkpoint.get('epoch')}, val_acc={checkpoint.get('val_acc', 'N/A')}")
            else:
                state_dict = checkpoint
                # Détecter depuis classifier.1.weight uniquement
                if 'classifier.1.weight' in state_dict:
                    num_classes = state_dict['classifier.1.weight'].shape[0]
                else:
                    num_classes = 22

            # Créer le modèle EfficientNet B4 avec TORCHVISION (le checkpoint a été entraîné avec torchvision)
            from torchvision.models import efficientnet_b0
            
            self.brand_model = efficientnet_b0(weights=None)
            self.brand_model.classifier[1] = torch.nn.Linear(
                self.brand_model.classifier[1].in_features, 
                num_classes
            )

            # Charger les poids
            self.brand_model.load_state_dict(state_dict)
            self.brand_model.to(self.device)
            self.brand_model.eval()
            
            # Transforms pour EfficientNet B4 (input 380x380)
            self.brand_transform = transforms.Compose([
                transforms.Resize((380, 380)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print(f"✅ EfficientNet: {model_path} ({num_classes} classes)")
            
        except Exception as e:
            print(f"⚠️ Erreur chargement EfficientNet: {e}")
            import traceback
            traceback.print_exc()
            self.brand_model = None
        
        def reload_model(self, model_name: str) -> Tuple[bool, str]:
            """
            Recharge un modèle YOLO plaque depuis models/.
            
            Args:
                model_name: Nom du fichier (ex: 'best.pt')
                
            Returns:
                (success, message)
            """
            try:
                # Construire le chemin
                if model_name.startswith('models/'):
                    model_path = model_name
                else:
                    model_path = f"models/{model_name}"
                
                if not os.path.exists(model_path):
                    return False, f"❌ Modèle non trouvé: {model_name}"
                
                print(f"📦 Chargement nouveau modèle: {model_name}")
                
                is_custom = os.path.basename(model_path) in CUSTOM_MODELS
                
                if is_custom and CUSTOM_YOLO_AVAILABLE:
                    new_model = load_custom_model(model_path, device=self.device)
                else:
                    new_model = YOLO(model_path)
                
                # Remplacer l'ancien modèle
                old_model = self.yolo_plate
                self.yolo_plate = new_model
                self.model_path = model_path
                self.is_custom_model = is_custom
                
                # Cleanup
                del old_model
                gc.collect()
                
                return True, f"✅ Modèle '{model_name}' chargé!"
                
            except Exception as e:
                return False, f"❌ Erreur: {e}"
        
        @staticmethod
        def get_available_models() -> List[str]:
            """Liste les modèles .pt disponibles dans models/."""
            models_dir = Path("models")
            if not models_dir.exists():
                return ["best.pt"]
            
            models = [f.name for f in models_dir.glob("*.pt")]
            models += [f.name for f in models_dir.glob("*.pth")]
            return sorted(models) if models else ["best.pt"]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BRANCHE 1: PLAQUE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_plates(self, image: np.ndarray, 
                      conf_threshold: float = 0.5) -> List[Dict]:
        """
        Détecte les plaques dans une image.
        
        Args:
            image: Image BGR ou RGB numpy
            conf_threshold: Seuil de confiance
            
        Returns:
            Liste de détections [{bbox, confidence, roi}, ...]
        """
        if self.yolo_plate is None:
            return []
        
        plates_data = []
        
        try:
            if self.is_custom_model:
                # Custom SimpleYOLO inference
                plates_data = self._run_custom_inference(image, conf_threshold)
            else:
                # Standard YOLOv8
                results = self.yolo_plate(image, conf=conf_threshold, verbose=False)
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0])
                        
                        # Extraire ROI
                        roi = image[y1:y2, x1:x2]
                        
                        plates_data.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'roi': roi
                        })
            
        except Exception as e:
            print(f"⚠️ Erreur détection plaque: {e}")
        
        return plates_data
    
    def _run_custom_inference(self, img: np.ndarray, conf_threshold: float) -> List[Dict]:
        """Inference avec SimpleYOLO custom."""
        if not TORCH_AVAILABLE:
            return []
        
        img_h, img_w = img.shape[:2]
        
        # Préprocessing
        img_resized = cv2.resize(img, (416, 416))
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 4:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGB)
        elif img_resized.shape[2] == 3:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Forward
        with torch.no_grad():
            output = self.yolo_plate(img_tensor)
        
        pred = output[0].cpu()
        GRID_SIZE = 13
        
        plates = []
        for cy in range(GRID_SIZE):
            for cx in range(GRID_SIZE):
                obj_conf = pred[cy, cx, 0].item()
                if obj_conf > conf_threshold:
                    xc = (cx + pred[cy, cx, 1].item()) / GRID_SIZE
                    yc = (cy + pred[cy, cx, 2].item()) / GRID_SIZE
                    w = abs(pred[cy, cx, 3].item())
                    h = abs(pred[cy, cx, 4].item())
                    
                    x1 = int((xc - w/2) * img_w)
                    y1 = int((yc - h/2) * img_h)
                    x2 = int((xc + w/2) * img_w)
                    y2 = int((yc + h/2) * img_h)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        roi = img[y1:y2, x1:x2]
                        plates.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': obj_conf,
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
            # Convertir en niveaux de gris (requis par fast-plate-ocr)
            if len(roi.shape) == 3:
                plate_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                plate_gray = roi
            
            # Appeler OCR SANS return_confidence
            predictions = self.ocr_model.run(plate_gray)
            
            if predictions and len(predictions) > 0:
                text = predictions[0].plate if hasattr(predictions[0], 'plate') else str(predictions[0])
                text = text.upper().replace(' ', '').replace('-', '')
                return text, 0.95  # Confiance fixe comme dans l'original
            
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
        
        return "", 0.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BRANCHE 2: VÉHICULE + MARQUE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_vehicle(self, image: np.ndarray, 
                       conf_threshold: float = 0.5) -> Optional[Dict]:
        """
        Détecte le véhicule principal dans une image.
        
        Args:
            image: Image BGR numpy
            conf_threshold: Seuil de confiance
            
        Returns:
            {bbox, confidence, class_name, roi} ou None
        """
        if self.yolo_vehicle is None:
            return None
        
        try:
            # Classes COCO pour véhicules
            vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
            
            results = self.yolo_vehicle(image, conf=conf_threshold, verbose=False)
            
            best_detection = None
            best_area = 0
            
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    
                    if class_id in vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        area = (x2 - x1) * (y2 - y1)
                        
                        if area > best_area:
                            best_area = area
                            roi = image[y1:y2, x1:x2]
                            
                            best_detection = {
                                'bbox': (x1, y1, x2, y2),
                                'confidence': float(box.conf[0]),
                                'class_id': class_id,
                                'class_name': vehicle_classes[class_id],
                                'roi': roi
                            }
            
            return best_detection
            
        except Exception as e:
            print(f"⚠️ Erreur détection véhicule: {e}")
            return None
    
    def classify_brand(self, vehicle_roi: np.ndarray) -> Optional[Dict]:
        """
        Classifie la marque d'un véhicule.
        
        Args:
            vehicle_roi: Image du véhicule (BGR)
            
        Returns:
            {brand, confidence, top3} ou None
        """
        if self.brand_model is None or self.brand_transform is None:
            return None
        
        try:
            # Convertir BGR → RGB → PIL
            if len(vehicle_roi.shape) == 3 and vehicle_roi.shape[2] == 3:
                vehicle_rgb = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2RGB)
            else:
                vehicle_rgb = vehicle_roi
            
            pil_image = Image.fromarray(vehicle_rgb)
            
            # Transform
            input_tensor = self.brand_transform(pil_image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.brand_model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
            
            # Top 3
            top3_probs, top3_indices = torch.topk(probs, min(3, len(probs)))
            
            brand_classes = self.model_config.brand_classes
            
            top3 = [
                (brand_classes[idx] if idx < len(brand_classes) else f"class_{idx}", 
                 float(prob))
                for idx, prob in zip(top3_indices.tolist(), top3_probs.tolist())
            ]
            
            best_idx = top3_indices[0].item()
            best_prob = top3_probs[0].item()
            best_brand = brand_classes[best_idx] if best_idx < len(brand_classes) else f"class_{best_idx}"
            
            return {
                'brand': best_brand,
                'confidence': best_prob,
                'top3': top3
            }
            
        except Exception as e:
            print(f"⚠️ Erreur classification marque: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PIPELINE PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    def process_image(self, image_path: Union[str, np.ndarray],
                      conf_threshold: float = 0.5,
                      run_branch2: bool = True,
                      declared_brand: str = None) -> Dict:
        """
        Traite une image à travers le pipeline complet.
        
        Args:
            image_path: Chemin ou image numpy BGR
            conf_threshold: Seuil de confiance YOLO
            run_branch2: Exécuter la branche 2 (véhicule + marque)
            declared_brand: Marque déclarée (pour mismatch detection)
            
        Returns:
            Dictionnaire avec tous les résultats
        """
        # Charger l'image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Impossible de charger: {image_path}")
        else:
            img = image_path.copy()
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Résultat (compatible format original)
        results = {
            'step1_raw': img_rgb.copy(),
            'step2_detection': None,
            'step3_roi': [],
            'step4_ocr': [],
            'step5_final': None,
            # Branche 2
            'vehicle_detection': None,
            'vehicle_crop': None,
            'brand_result': None,
            'mismatch': False,
            'declared_brand': declared_brand,
            # Métadonnées
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'image_size': img.shape[:2],
                'detections': [],
                'conditions': self._estimate_conditions(img_rgb)
            }
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # BRANCHE 1: Plaque + OCR
        # ─────────────────────────────────────────────────────────────────────
        
        # Step 2: Détection plaques
        plates_data = self.detect_plates(img, conf_threshold)
        
        img_detected = img_rgb.copy()
        for plate in plates_data:
            x1, y1, x2, y2 = plate['bbox']
            cv2.rectangle(img_detected, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        results['step2_detection'] = img_detected
        results['metadata']['detections'] = [
            {'bbox': p['bbox'], 'confidence': p['confidence']} for p in plates_data
        ]
        
        # Step 3 & 4: ROI + OCR
        final_img = img_rgb.copy()
        
        for plate in plates_data:
            roi = plate['roi']
            x1, y1, x2, y2 = plate['bbox']
            
            results['step3_roi'].append(roi)
            
            # OCR
            text, ocr_conf = self.read_plate(roi)
            
            results['step4_ocr'].append({
                'text': text,
                'confidence': ocr_conf,
                'detection_confidence': plate['confidence'],
                'bbox': plate['bbox']
            })
            
            # Annoter image finale
            cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            label = text if text else "???"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
            cv2.rectangle(final_img, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
            cv2.putText(final_img, label, (x1 + 5, y1 - 5), font, 0.8, (0, 0, 0), 2)
        
        results['step5_final'] = final_img
        
        # ─────────────────────────────────────────────────────────────────────
        # BRANCHE 2: Véhicule + Marque
        # ─────────────────────────────────────────────────────────────────────
        
        if run_branch2 and self.yolo_vehicle is not None:
            vehicle = self.detect_vehicle(img)
            
            if vehicle:
                results['vehicle_detection'] = {
                    'bbox': vehicle['bbox'],
                    'confidence': vehicle['confidence'],
                    'class_name': vehicle['class_name']
                }
                results['vehicle_crop'] = vehicle['roi']
                
                # Classification marque
                if self.brand_model is not None and vehicle['roi'].size > 0:
                    brand_result = self.classify_brand(vehicle['roi'])
                    
                    if brand_result:
                        results['brand_result'] = brand_result
                        
                        # Mismatch detection
                        if declared_brand:
                            predicted = brand_result['brand'].lower()
                            declared = declared_brand.lower()
                            results['mismatch'] = predicted != declared
                        
                        # Annoter image finale avec marque
                        x1, y1, x2, y2 = vehicle['bbox']
                        brand_label = f"{brand_result['brand']} ({brand_result['confidence']:.0%})"
                        color = (255, 0, 0) if results['mismatch'] else (255, 165, 0)
                        cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(final_img, brand_label, (x1, y2 + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        results['step5_final'] = final_img
        
        return results
    
    def _estimate_conditions(self, img_rgb: np.ndarray) -> Dict:
        """Estime les conditions de l'image (luminosité, netteté)."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 60:
            lighting = "Night / Very low light"
            lighting_emoji = "🌙"
        elif mean_brightness < 100:
            lighting = "Low light"
            lighting_emoji = "🌆"
        elif mean_brightness < 160:
            lighting = "Medium light"
            lighting_emoji = "☁️"
        else:
            lighting = "Daylight / Well lit"
            lighting_emoji = "☀️"
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            blur = "High blur"
        elif laplacian_var < 500:
            blur = "Medium blur"
        else:
            blur = "Sharp"
        
        return {
            'brightness': float(mean_brightness),
            'lighting': lighting,
            'lighting_emoji': lighting_emoji,
            'blur': blur,
            'blur_score': float(laplacian_var)
        }
    
    def process_video(self, video_path: str, max_frames: int = 30,
                      conf_threshold: float = 0.5) -> List[Dict]:
        """
        Traite les frames d'une vidéo.
        
        Args:
            video_path: Chemin vidéo
            max_frames: Nombre max de frames
            conf_threshold: Seuil de confiance
            
        Returns:
            Liste de résultats par frame
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        results = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_result = self.process_image(frame, conf_threshold)
            frame_result['frame_number'] = int(idx)
            frame_result['timestamp'] = idx / cap.get(cv2.CAP_PROP_FPS)
            results.append(frame_result)
        
        cap.release()
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Test ALPRPipeline...")
    
    pipeline = ALPRPipeline(load_branch2=True)
    
    print(f"\nModèles disponibles: {pipeline.get_available_models()}")
    print(f"Modèle plaque chargé: {pipeline.model_path}")
    print(f"YOLO Plaque: {'✅' if pipeline.yolo_plate else '❌'}")
    print(f"YOLO Véhicule: {'✅' if pipeline.yolo_vehicle else '❌'}")
    print(f"OCR: {'✅' if pipeline.ocr_model else '❌'}")
    print(f"EfficientNet: {'✅' if pipeline.brand_model else '❌'}")
    
    # Test avec image grise
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (128, 128, 128)
    
    result = pipeline.process_image(test_img, conf_threshold=0.3)
    print(f"\nTest image grise:")
    print(f"  Détections plaque: {len(result['metadata']['detections'])}")
    print(f"  Véhicule détecté: {result['vehicle_detection'] is not None}")
    print(f"  Marque: {result['brand_result']}")
    
    print("\n✅ Tests OK")
