"""
ALPR Engine - Adaptateur Gradio du cœur d'inférence.

Façade de compatibilité : elle conserve l'interface historique `ALPRPipeline`
(dictionnaire « step1_raw ... step5_final ») utilisée par app.py et
video_processor.py, mais toute l'inférence est déléguée à `src.inference`.
Elle ne contient plus aucune logique de chargement ni de prédiction.

À terme (couche de services), app.py consommera directement `InferenceResult`
et cette façade disparaîtra.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.inference import ALPREngine, InferenceResult, is_brand_mismatch, load_image
from src.inference.rendering import annotate, draw_plate_boxes

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def _to_rgb(image_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image_bgr is None or image_bgr.ndim != 3:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class ALPRPipeline:
    """
    Pipeline ALPR pour l'interface Gradio (façade du cœur d'inférence).

    Branche 1 : détection plaque + OCR. Branche 2 : véhicule + marque.
    """

    def __init__(self, model_path: str = None, load_branch2: bool = True,
                 engine: Optional[ALPREngine] = None):
        self.engine = engine or ALPREngine(load_branch2=load_branch2, plate_model_path=model_path)
        self.settings = self.engine.settings

    # ═══════════════════════════════════════════════════════════════════════════
    # ÉTAT DES MODÈLES (lu par l'onglet Settings)
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def device(self) -> str:
        return self.engine.models.device

    @property
    def model_path(self) -> Optional[str]:
        return self.engine.models.sources.get("plate_detector")

    @property
    def yolo_plate(self):
        return self.engine.models.plate_detector

    @property
    def ocr_model(self):
        return self.engine.models.plate_reader

    @property
    def yolo_vehicle(self):
        return self.engine.models.vehicle_detector

    @property
    def brand_model(self):
        return self.engine.models.brand_classifier

    def reload_model(self, model_name: str) -> Tuple[bool, str]:
        return self.engine.reload_plate_detector(model_name)

    @staticmethod
    def get_available_models() -> List[str]:
        """Liste les modèles .pt / .pth disponibles dans models/."""
        if not MODELS_DIR.exists():
            return ["best.pt"]
        models = [f.name for f in MODELS_DIR.glob("*.pt")]
        models += [f.name for f in MODELS_DIR.glob("*.pth")]
        return sorted(models) if models else ["best.pt"]

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAITEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def process_image(self, image_path: Union[str, np.ndarray],
                      conf_threshold: Optional[float] = None,
                      run_branch2: bool = True,
                      declared_brand: str = None) -> Dict:
        """
        Traite une image et renvoie le dictionnaire attendu par l'interface.

        Toutes les images renvoyées sont en RGB.
        """
        img = load_image(image_path)
        result = self.engine.analyze(img, conf_threshold=conf_threshold, run_branch2=run_branch2)
        return self.to_legacy_dict(img, result, declared_brand)

    @staticmethod
    def to_legacy_dict(img_bgr: np.ndarray, result: InferenceResult,
                       declared_brand: Optional[str] = None) -> Dict:
        """Convertit un `InferenceResult` au format historique de l'interface."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mismatch = is_brand_mismatch(result.brand.brand if result.brand else None, declared_brand)

        return {
            'step1_raw': img_rgb,
            'step2_detection': draw_plate_boxes(img_rgb, result),
            'step3_roi': [_to_rgb(p.crop) for p in result.plates],
            'step4_ocr': [
                {
                    'text': p.text,
                    # L'interface attend un nombre : 0.0 quand l'OCR ne fournit pas de confiance
                    'confidence': p.ocr_confidence if p.ocr_confidence is not None else 0.0,
                    'detection_confidence': p.detection_confidence,
                    'bbox': p.bbox,
                }
                for p in result.plates
            ],
            'step5_final': annotate(img_rgb, result, mismatch=mismatch),
            'vehicle_detection': result.vehicle.to_dict() if result.vehicle else None,
            'vehicle_crop': _to_rgb(result.vehicle.crop) if result.vehicle else None,
            'brand_result': (
                {'brand': result.brand.brand, 'confidence': result.brand.confidence,
                 'top3': list(result.brand.top3)}
                if result.brand else None
            ),
            'mismatch': mismatch,
            'declared_brand': declared_brand,
            'metadata': {
                'timestamp': result.timestamp,
                'image_size': result.image_size,
                'detections': [
                    {'bbox': p.bbox, 'confidence': p.detection_confidence} for p in result.plates
                ],
                'conditions': result.conditions.to_dict(),
                'inference_time_s': result.inference_time_s,
            },
        }

    def process_video(self, video_path: str, max_frames: int = 30,
                      conf_threshold: Optional[float] = None) -> List[Dict]:
        """Traite un échantillon de frames d'une vidéo."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        indices = np.linspace(0, max(total_frames - 1, 0), min(max_frames, total_frames), dtype=int)

        results = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frame_result = self.process_image(frame, conf_threshold)
            frame_result['frame_number'] = int(idx)
            frame_result['timestamp'] = idx / fps
            results.append(frame_result)
        cap.release()
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pipeline = ALPRPipeline(load_branch2=True)
    print(f"Modèles disponibles : {pipeline.get_available_models()}")
    for role, info in pipeline.engine.status().items():
        print(f"  {role}: {info}")
