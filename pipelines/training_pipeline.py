"""
ALPR Engine - Training Pipeline
Pipeline d'entraînement pour le modèle EfficientNet de classification de marques.
Peut être appelé localement ou via Airflow.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import tempfile
import shutil

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import get_settings

# Imports optionnels
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TrainingPipeline:
    """
    Pipeline d'entraînement EfficientNet.
    
    Étapes:
    1. Préparation du dataset (depuis S3/local)
    2. Split train/val/test
    3. Entraînement avec early stopping
    4. Évaluation
    5. Log vers MLflow
    6. Promotion si meilleur que production
    """
    
    def __init__(self):
        """Initialise le pipeline."""
        self.settings = get_settings()
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        # Config entraînement
        self.config = {
            'model_name': 'efficientnet_b4',
            'batch_size': 32,
            'epochs': 20,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'early_stopping_patience': 5,
            'min_accuracy_improvement': 0.01,
        }
        
        print(f"✅ TrainingPipeline initialisé (device: {self.device})")
    
    def check_requirements(self) -> Tuple[bool, str]:
        """Vérifie que toutes les dépendances sont disponibles."""
        if not TORCH_AVAILABLE:
            return False, "PyTorch non installé"
        if not TIMM_AVAILABLE:
            return False, "timm non installé"
        if not SKLEARN_AVAILABLE:
            return False, "scikit-learn non installé"
        return True, "OK"
    
    def prepare_dataset(self, source_dir: str = None) -> Dict:
        """
        Prépare le dataset pour l'entraînement.
        
        Args:
            source_dir: Dossier source des images labellisées
            
        Returns:
            Dict avec chemins train/val/test et métadonnées
        """
        from src.data.storage import get_storage
        
        storage = get_storage()
        
        # Créer un dossier temporaire
        work_dir = Path(tempfile.mkdtemp(prefix="alpr_training_"))
        train_dir = work_dir / "train"
        val_dir = work_dir / "val"
        test_dir = work_dir / "test"
        
        print(f"📦 Préparation du dataset dans {work_dir}")
        
        # Récupérer les images depuis S3/local
        if source_dir:
            source_path = Path(source_dir)
        else:
            # Télécharger depuis S3
            files = storage.list_files("labeled/", bucket_type='dataset', max_keys=10000)
            source_path = work_dir / "source"
            source_path.mkdir()
            
            for f in files:
                dest = source_path / f.key.replace("labeled/", "")
                dest.parent.mkdir(parents=True, exist_ok=True)
                storage.download_file(f.key, dest, bucket_type='dataset')
        
        # Organiser par classe
        by_class = {}
        for class_dir in source_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                by_class[class_name] = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        print(f"   {len(by_class)} classes trouvées")
        
        # Split train/val/test (70/15/15)
        stats = {'train': 0, 'val': 0, 'test': 0}
        
        for class_name, class_files in by_class.items():
            (train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (val_dir / class_name).mkdir(parents=True, exist_ok=True)
            (test_dir / class_name).mkdir(parents=True, exist_ok=True)
            
            if len(class_files) < 3:
                train_files, val_files, test_files = class_files, [], []
            else:
                train_files, temp_files = train_test_split(class_files, test_size=0.3, random_state=42)
                if len(temp_files) >= 2:
                    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
                else:
                    val_files, test_files = temp_files, []
            
            for f in train_files:
                shutil.copy(f, train_dir / class_name / f.name)
                stats['train'] += 1
            for f in val_files:
                shutil.copy(f, val_dir / class_name / f.name)
                stats['val'] += 1
            for f in test_files:
                shutil.copy(f, test_dir / class_name / f.name)
                stats['test'] += 1
        
        print(f"   Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
        
        return {
            'work_dir': str(work_dir),
            'train_dir': str(train_dir),
            'val_dir': str(val_dir),
            'test_dir': str(test_dir),
            'num_classes': len(by_class),
            'class_names': list(by_class.keys()),
            'stats': stats
        }
    
    def train(self, dataset_info: Dict) -> Dict:
        """
        Entraîne le modèle EfficientNet.
        
        Args:
            dataset_info: Info du dataset (depuis prepare_dataset)
            
        Returns:
            Dict avec métriques et chemin du modèle
        """
        from src.models.mlflow_client import get_mlflow_manager, MLflowManager
        
        train_dir = Path(dataset_info['train_dir'])
        val_dir = Path(dataset_info['val_dir'])
        num_classes = dataset_info['num_classes']
        class_names = dataset_info['class_names']
        
        print(f"🚀 Entraînement EfficientNet B4 ({num_classes} classes)")
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Datasets
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'],
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'],
                                shuffle=False, num_workers=4, pin_memory=True)
        
        # Modèle
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        model = model.to(self.device)
        
        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'],
                                weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        
        # MLflow
        mlflow_manager = get_mlflow_manager()
        run_id = mlflow_manager.start_run(
            MLflowManager.EXPERIMENT_BRAND_CLASSIFICATION,
            run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={'pipeline': 'training_pipeline', 'num_classes': str(num_classes)}
        )
        
        mlflow_manager.log_params(self.config)
        mlflow_manager.log_params({'num_classes': num_classes, 'train_size': len(train_dataset)})
        
        # Training loop
        best_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Train
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_accuracy = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_accuracy = val_correct / val_total
            
            print(f"   Epoch {epoch+1}/{self.config['epochs']}: "
                  f"Train={train_accuracy:.4f}, Val={val_accuracy:.4f}")
            
            # Log metrics
            mlflow_manager.log_metrics({
                'train_loss': train_loss / len(train_loader),
                'train_accuracy': train_accuracy,
                'val_loss': val_loss / len(val_loader),
                'val_accuracy': val_accuracy,
            }, step=epoch)
            
            scheduler.step(val_accuracy)
            
            # Best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"   Early stopping at epoch {epoch+1}")
                break
        
        # Sauvegarder
        model.load_state_dict(best_model_state)
        model_path = Path(dataset_info['work_dir']) / "best_model.pth"
        torch.save(best_model_state, model_path)
        
        # Log model
        mlflow_manager.log_metrics({'best_accuracy': best_accuracy})
        mlflow_manager.log_artifact(str(model_path))
        model_uri = mlflow_manager.log_model(model, "efficientnet_brand_classifier",
                                              registered_name=MLflowManager.MODEL_EFFICIENTNET)
        
        mlflow_manager.end_run()
        
        print(f"✅ Entraînement terminé: accuracy={best_accuracy:.4f}")
        
        return {
            'run_id': run_id,
            'model_path': str(model_path),
            'model_uri': model_uri,
            'best_accuracy': best_accuracy,
            'num_classes': num_classes,
            'class_names': class_names
        }
    
    def evaluate_and_promote(self, training_result: Dict) -> Tuple[bool, str]:
        """
        Évalue et promeut le modèle si meilleur que production.
        
        Args:
            training_result: Résultat de train()
            
        Returns:
            (promoted, message)
        """
        from src.models.mlflow_client import get_mlflow_manager, MLflowManager
        
        mlflow_manager = get_mlflow_manager()
        new_accuracy = training_result['best_accuracy']
        
        # Comparer avec production
        production_model = mlflow_manager.get_production_model(MLflowManager.MODEL_EFFICIENTNET)
        
        if production_model is None:
            # Pas de modèle en prod, promouvoir directement
            versions = mlflow_manager.get_model_versions(MLflowManager.MODEL_EFFICIENTNET)
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                mlflow_manager.promote_to_production(MLflowManager.MODEL_EFFICIENTNET, latest.version)
                return True, f"✅ Premier modèle promu en production (accuracy={new_accuracy:.4f})"
            return False, "❌ Aucune version trouvée"
        
        prod_accuracy = production_model.accuracy
        improvement = new_accuracy - prod_accuracy
        
        if improvement >= self.config['min_accuracy_improvement']:
            versions = mlflow_manager.get_model_versions(MLflowManager.MODEL_EFFICIENTNET)
            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            mlflow_manager.promote_to_production(MLflowManager.MODEL_EFFICIENTNET, latest.version)
            return True, f"✅ Modèle promu: {prod_accuracy:.4f} → {new_accuracy:.4f} (+{improvement:.4f})"
        
        return False, f"⏭️ Amélioration insuffisante: {improvement:+.4f} (min={self.config['min_accuracy_improvement']})"
    
    def run(self, source_dir: str = None, cleanup: bool = True) -> Dict:
        """
        Exécute le pipeline complet.
        
        Args:
            source_dir: Dossier source (optionnel)
            cleanup: Supprimer les fichiers temporaires
            
        Returns:
            Dict avec résultats
        """
        # Check requirements
        ok, msg = self.check_requirements()
        if not ok:
            return {'success': False, 'error': msg}
        
        try:
            # Prepare
            dataset_info = self.prepare_dataset(source_dir)
            
            # Train
            training_result = self.train(dataset_info)
            
            # Evaluate & promote
            promoted, promotion_msg = self.evaluate_and_promote(training_result)
            
            # Cleanup
            if cleanup:
                shutil.rmtree(dataset_info['work_dir'])
            
            return {
                'success': True,
                'accuracy': training_result['best_accuracy'],
                'promoted': promoted,
                'message': promotion_msg,
                'run_id': training_result['run_id']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ALPR Training Pipeline")
    parser.add_argument("--source", help="Dossier source des images")
    parser.add_argument("--no-cleanup", action="store_true", help="Garder les fichiers temp")
    args = parser.parse_args()
    
    pipeline = TrainingPipeline()
    result = pipeline.run(source_dir=args.source, cleanup=not args.no_cleanup)
    
    print(f"\n{'='*50}")
    print(f"Résultat: {'SUCCESS' if result['success'] else 'FAILED'}")
    if result['success']:
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Promu: {result['promoted']}")
        print(f"Message: {result['message']}")
    else:
        print(f"Erreur: {result.get('error', 'Unknown')}")
