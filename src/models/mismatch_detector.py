"""
ALPR Engine - Mismatch Detector
Détection et gestion des erreurs de classification marque.
Alimente la boucle d'amélioration continue (retraining EfficientNet).
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import get_settings
from src.inference.brands import is_brand_mismatch, normalize_brand

# Imports optionnels
try:
    import numpy as np
    from PIL import Image
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from src.data.database import DatabaseManager
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    from src.data.storage import get_storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False


@dataclass
class MismatchRecord:
    """Enregistrement d'un mismatch détecté."""
    plaque: str
    marque_predite: str
    marque_declaree: str
    confiance: float
    timestamp: str
    image_path: Optional[str] = None
    log_id: Optional[int] = None
    status: str = "pending"  # pending, validated, rejected, exported
    
    def to_dict(self) -> Dict:
        return {
            'plaque': self.plaque,
            'marque_predite': self.marque_predite,
            'marque_declaree': self.marque_declaree,
            'confiance': self.confiance,
            'timestamp': self.timestamp,
            'image_path': self.image_path,
            'log_id': self.log_id,
            'status': self.status
        }


class MismatchDetector:
    """
    Détecteur et gestionnaire de mismatches marque.
    
    Workflow:
    1. Détection: marque_predite ≠ marque_declaree
    2. Stockage: image → S3, log → DB
    3. File d'attente: pour validation opérateur
    4. Export: vers dataset labellisé pour retraining
    """
    
    def __init__(self, db=None, storage=None):
        """
        Initialise le détecteur de mismatch.

        `db` et `storage` peuvent être fournis par le conteneur de services,
        pour partager une seule connexion ; à défaut, ils sont créés ici.
        """
        self.settings = get_settings()
        
        # Database
        self.db = db if db is not None else (DatabaseManager() if DB_AVAILABLE else None)
        
        # Storage
        self.storage = storage if storage is not None else (get_storage() if STORAGE_AVAILABLE else None)
        
        # File d'attente en mémoire (backup si DB non dispo)
        self._queue: List[MismatchRecord] = []

        # IDs des logs déjà traités (validés/rejetés) dans cette session
        self._validated_ids: set = set()

        # Stats
        self._stats = {
            'total_detected': 0,
            'pending': 0,
            'validated': 0,
            'rejected': 0,
            'exported': 0
        }
    def _normalize_brand(self, brand: str) -> str:
        """Nom de marque normalisé (règle unique du cœur d'inférence)."""
        return normalize_brand(brand)
        print("✅ MismatchDetector initialisé")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def check_mismatch(self, plaque: str, marque_predite: str,
                       marque_declaree: str, confiance: float = 0.0) -> bool:
        """
        Vérifie s'il y a un mismatch entre marque prédite et déclarée.
        
        Args:
            plaque: Plaque d'immatriculation
            marque_predite: Marque prédite par EfficientNet
            marque_declaree: Marque déclarée dans la DB
            confiance: Score de confiance de la prédiction
            
        Returns:
            True si mismatch détecté
        """
        return is_brand_mismatch(marque_predite, marque_declaree)
    
    def detect_and_record(self, plaque: str, marque_predite: str,
                          marque_declaree: str, confiance: float,
                          vehicle_crop: 'np.ndarray' = None,
                          log_to_db: bool = True) -> Optional[MismatchRecord]:
        """
        Détecte un mismatch et l'enregistre.
        
        Args:
            plaque: Plaque d'immatriculation
            marque_predite: Marque prédite
            marque_declaree: Marque déclarée
            confiance: Score de confiance
            vehicle_crop: Image du véhicule (numpy array RGB)
            log_to_db: Enregistrer un log en base. False quand l'appelant
                (ALPRService) a déjà enregistré le passage.
            
        Returns:
            MismatchRecord si mismatch détecté, None sinon
        """
        # Vérifier le mismatch
        if not self.check_mismatch(plaque, marque_predite, marque_declaree, confiance):
            return None
        
        timestamp = datetime.now().isoformat()
        image_key = None   # clé de l'image dans le bucket « captures » (pas une URL)
        log_id = None
        
        # Sauvegarder l'image vers S3/local
        if vehicle_crop is not None and self.storage and NUMPY_AVAILABLE:
            try:
                # Générer le chemin
                date_str = datetime.now().strftime("%Y%m%d")
                safe_plaque = plaque.replace("-", "").replace(" ", "")
                key = f"mismatches/{date_str}/{safe_plaque}_{datetime.now().strftime('%H%M%S')}.jpg"
                
                # Upload
                success, result = self.storage.upload_image(
                    vehicle_crop, key, bucket_type='captures'
                )
                
                if success:
                    image_key = key
                else:
                    print(f"⚠️ Image mismatch non sauvegardée: {result}")
                    
            except Exception as e:
                print(f"⚠️ Erreur sauvegarde image mismatch: {e}")
        
        # Logger dans la DB
        if self.db and log_to_db:
            try:
                self.db.add_log(
                    plaque=plaque,
                    authorized=True,  # On assume accès autorisé malgré mismatch
                    normalized=plaque.replace("-", "").replace(" ", ""),
                    marque_predite=marque_predite,
                    marque_confiance=confiance,
                    mismatch=True
                )
            except Exception as e:
                print(f"⚠️ Erreur log DB mismatch: {e}")
        
        # Créer le record
        record = MismatchRecord(
            plaque=plaque,
            marque_predite=marque_predite,
            marque_declaree=marque_declaree,
            confiance=confiance,
            timestamp=timestamp,
            image_path=image_key,
            log_id=log_id,
            status="pending"
        )
        
        # Ajouter à la file
        self._queue.append(record)
        self._stats['total_detected'] += 1
        self._stats['pending'] += 1
        
        print(f"⚠️ Mismatch détecté: {plaque} → prédit={marque_predite}, déclaré={marque_declaree}")
        
        return record
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FILE D'ATTENTE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_pending(self, limit: int = 50) -> List[MismatchRecord]:
        """
        Récupère les mismatches en attente de validation.
        
        Args:
            limit: Nombre maximum de records
            
        Returns:
            Liste de MismatchRecord
        """
        # Depuis la DB si disponible
        if self.db:
            try:
                logs = self.db.get_mismatch_logs(limit=limit)
                records = []
                for log in logs:
                    if log['id'] in self._validated_ids:
                        continue
                    # Récupérer la marque déclarée depuis residents
                    resident = self.db.get_plate_with_brand(log['plaque'])
                    declared = resident.get('marque_declaree', '') if resident else ''

                    records.append(MismatchRecord(
                        plaque=log['plaque'],
                        marque_predite=log.get('marque_predite', ''),
                        marque_declaree=declared,
                        confiance=log.get('marque_confiance', 0.0) or 0.0,
                        timestamp=log['timestamp'],
                        image_path=log.get('image_key'),
                        log_id=log['id'],
                        status="pending"
                    ))
                return records
            except Exception as e:
                print(f"⚠️ Erreur récupération mismatches DB: {e}")
        
        # Fallback: file en mémoire
        return [r for r in self._queue if r.status == "pending"][:limit]
    
    def get_queue_size(self) -> int:
        """Retourne le nombre de mismatches en attente."""
        if self.db:
            try:
                stats = self.db.get_statistics()
                return stats.get('mismatches_pending', 0)
            except:
                pass
        
        return len([r for r in self._queue if r.status == "pending"])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def validate(self, record: MismatchRecord, 
                 correct_brand: str,
                 validator: str = "operator") -> Tuple[bool, str]:
        """
        Valide un mismatch avec la marque correcte.
        
        Args:
            record: Le MismatchRecord à valider
            correct_brand: La marque correcte (peut être predicted ou declared ou autre)
            validator: Identifiant du validateur
            
        Returns:
            (success, message)
        """
        label = normalize_brand(correct_brand)
        try:
            record.status = "validated"
            self._mark_in_db(record, "validated", label)
            self._stats['pending'] -= 1
            self._stats['validated'] += 1

            if not (record.image_path and self.storage):
                return True, f"✅ Validé (sans image): {record.plaque} → {label}"

            # Copier l'image de l'incohérence (bucket captures) vers le dataset labellisé
            found, image = self.storage.download_image(record.image_path, bucket_type='captures')
            if not found:
                return True, f"⚠️ Validé: {record.plaque} → {label}, image introuvable ({image})"

            success, result = self.storage.save_labeled_image(
                image=image.convert('RGB'),
                label=label,
                plaque=record.plaque,
                confidence=record.confiance,
                metadata={
                    'original_prediction': record.marque_predite,
                    'declared': record.marque_declaree,
                    'validated_by': validator,
                    'validated_at': datetime.now().isoformat()
                }
            )
            if success:
                return True, f"✅ Validé: {record.plaque} → {label} (image ajoutée au jeu de données)"
            return True, f"⚠️ Validé: {record.plaque} → {label}, image non enregistrée ({result})"
            
        except Exception as e:
            return False, f"❌ Erreur validation: {e}"
    
    def reject(self, record: MismatchRecord, reason: str = "") -> Tuple[bool, str]:
        """
        Rejette un mismatch (faux positif, image inutilisable, etc.)
        
        Args:
            record: Le MismatchRecord à rejeter
            reason: Raison du rejet
            
        Returns:
            (success, message)
        """
        try:
            record.status = "rejected"
            self._mark_in_db(record, "rejected")
            self._stats['pending'] -= 1
            self._stats['rejected'] += 1

            return True, f"✅ Rejeté: {record.plaque} ({reason})"
            
        except Exception as e:
            return False, f"❌ Erreur rejet: {e}"
    
    def _mark_in_db(self, record: MismatchRecord, status: str, label: str = None) -> None:
        """Persiste la décision en base : l'incohérence quitte la file, même après redémarrage."""
        if record.log_id is not None:
            self._validated_ids.add(record.log_id)
        if self.db is not None and record.log_id is not None:
            try:
                self.db.set_log_validation(record.log_id, status, label)
            except Exception as e:
                print(f"⚠️ Décision non enregistrée en base: {e}")

    def validate_as_predicted(self, record: MismatchRecord,
                              validator: str = "operator") -> Tuple[bool, str]:
        """
        Valide avec la marque prédite (le modèle avait raison, DB incorrecte).
        
        Args:
            record: Le MismatchRecord
            validator: Identifiant du validateur
            
        Returns:
            (success, message)
        """
        return self.validate(record, record.marque_predite, validator)
    
    def validate_as_declared(self, record: MismatchRecord,
                             validator: str = "operator") -> Tuple[bool, str]:
        """
        Valide avec la marque déclarée (le modèle s'est trompé).
        
        Args:
            record: Le MismatchRecord
            validator: Identifiant du validateur
            
        Returns:
            (success, message)
        """
        return self.validate(record, record.marque_declaree, validator)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXPORT DATASET
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_dataset_stats(self) -> Dict:
        """
        Récupère les stats du dataset labellisé.
        
        Returns:
            Dict avec stats par marque
        """
        if self.storage:
            return self.storage.get_dataset_stats()
        
        return {'total': 0, 'by_label': {}, 'size_mb': 0.0}
    
    def is_ready_for_retraining(self, min_images: int = None) -> bool:
        """
        Vérifie si le dataset est prêt pour retraining.
        
        Args:
            min_images: Seuil minimum d'images (défaut: config)
            
        Returns:
            True si prêt
        """
        if min_images is None:
            min_images = self.settings.airflow.min_dataset_size
        
        stats = self.get_dataset_stats()
        return stats.get('total', 0) >= min_images
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du détecteur."""
        stats = self._stats.copy()
        stats['queue_size'] = self.get_queue_size()
        stats['dataset'] = self.get_dataset_stats()
        stats['ready_for_retraining'] = self.is_ready_for_retraining()
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_detector_instance: Optional[MismatchDetector] = None

def get_mismatch_detector() -> MismatchDetector:
    """Singleton pour obtenir le détecteur de mismatch."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MismatchDetector()
    return _detector_instance


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Test MismatchDetector...")
    
    detector = get_mismatch_detector()
    
    # Test détection
    is_mismatch = detector.check_mismatch(
        plaque="AB-123-CD",
        marque_predite="Peugeot",
        marque_declaree="Renault",
        confiance=0.85
    )
    print(f"Mismatch Peugeot vs Renault: {is_mismatch}")
    assert is_mismatch == True
    
    # Test pas de mismatch
    is_mismatch = detector.check_mismatch(
        plaque="AB-123-CD",
        marque_predite="Renault",
        marque_declaree="renault",  # Case différente
        confiance=0.90
    )
    print(f"Mismatch Renault vs renault: {is_mismatch}")
    assert is_mismatch == False
    
    # Test variantes
    is_mismatch = detector.check_mismatch(
        plaque="AB-123-CD",
        marque_predite="VW",
        marque_declaree="Volkswagen",
        confiance=0.75
    )
    print(f"Mismatch VW vs Volkswagen: {is_mismatch}")
    assert is_mismatch == False
    
    # Test record
    record = detector.detect_and_record(
        plaque="TEST-999-ZZ",
        marque_predite="BMW",
        marque_declaree="Audi",
        confiance=0.78,
        vehicle_crop=None  # Pas d'image pour le test
    )
    print(f"Record créé: {record}")
    
    # Stats
    stats = detector.get_stats()
    print(f"Stats: {stats}")
    
    print("✅ Tests OK")
