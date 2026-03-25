"""
ALPR Engine - MLflow Client
Tracking des expériences et gestion des modèles.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import get_settings

# MLflow import
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ mlflow non installé")


@dataclass
class ModelVersion:
    """Représente une version de modèle."""
    name: str
    version: str
    run_id: str
    stage: str  # None, Staging, Production, Archived
    accuracy: float
    created_at: str
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'version': self.version,
            'run_id': self.run_id,
            'stage': self.stage,
            'accuracy': self.accuracy,
            'created_at': self.created_at
        }


class MLflowManager:
    """
    Gestionnaire MLflow pour ALPR Engine.
    
    Fonctionnalités:
    - Tracking des runs d'entraînement
    - Logging des métriques et paramètres
    - Gestion du model registry
    - Promotion staging → production
    """
    
    # Noms des expériences
    EXPERIMENT_PLATE_DETECTION = "alpr-plate-detection"
    EXPERIMENT_BRAND_CLASSIFICATION = "alpr-brand-classification"
    
    # Noms des modèles dans le registry
    MODEL_EFFICIENTNET = "efficientnet-brand-classifier"
    MODEL_YOLO_PLATE = "yolo-plate-detector"
    
    def __init__(self):
        """Initialise le client MLflow."""
        self.settings = get_settings()
        self.mlflow_config = self.settings.mlflow
        
        if not MLFLOW_AVAILABLE:
            print("⚠️ MLflow non disponible")
            self.client = None
            return
        
        # Configurer l'URI
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        
        # Client pour API directe
        self.client = MlflowClient()
        
        # Créer les expériences si nécessaire
        self._ensure_experiments()
        
        print(f"✅ MLflow: {self.mlflow_config.tracking_uri}")
    
    def _ensure_experiments(self):
        """Crée les expériences si elles n'existent pas."""
        experiments = [
            self.EXPERIMENT_PLATE_DETECTION,
            self.EXPERIMENT_BRAND_CLASSIFICATION
        ]
        
        for exp_name in experiments:
            try:
                exp = mlflow.get_experiment_by_name(exp_name)
                if exp is None:
                    mlflow.create_experiment(exp_name)
                    print(f"  📁 Expérience créée: {exp_name}")
            except Exception as e:
                print(f"  ⚠️ Erreur création expérience {exp_name}: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRACKING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def start_run(self, experiment_name: str, run_name: str = None,
                  tags: Dict = None) -> Optional[str]:
        """
        Démarre un nouveau run MLflow.
        
        Args:
            experiment_name: Nom de l'expérience
            run_name: Nom du run (optionnel)
            tags: Tags additionnels
            
        Returns:
            Run ID ou None si erreur
        """
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            mlflow.set_experiment(experiment_name)
            
            run = mlflow.start_run(run_name=run_name)
            
            # Tags par défaut
            mlflow.set_tag("project", "alpr-engine")
            mlflow.set_tag("environment", self.settings.env)
            
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, str(value))
            
            return run.info.run_id
            
        except Exception as e:
            print(f"⚠️ Erreur démarrage run: {e}")
            return None
    
    def end_run(self, status: str = "FINISHED"):
        """Termine le run actif."""
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run(status=status)
            except:
                pass
    
    def log_params(self, params: Dict):
        """Log des paramètres."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            print(f"⚠️ Erreur log params: {e}")
    
    def log_metrics(self, metrics: Dict, step: int = None):
        """Log des métriques."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            for key, value in metrics.items():
                if step is not None:
                    mlflow.log_metric(key, value, step=step)
                else:
                    mlflow.log_metric(key, value)
        except Exception as e:
            print(f"⚠️ Erreur log metrics: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log un artifact (fichier)."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f"⚠️ Erreur log artifact: {e}")
    
    def log_model(self, model, model_name: str, signature = None,
                  registered_name: str = None) -> Optional[str]:
        """
        Log un modèle PyTorch.
        
        Args:
            model: Modèle PyTorch
            model_name: Nom pour l'artifact
            signature: Signature MLflow (optionnel)
            registered_name: Nom pour le model registry (optionnel)
            
        Returns:
            URI du modèle ou None
        """
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            # Log le modèle
            model_info = mlflow.pytorch.log_model(
                model,
                model_name,
                signature=signature,
                registered_model_name=registered_name
            )
            
            return model_info.model_uri
            
        except Exception as e:
            print(f"⚠️ Erreur log model: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL REGISTRY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def register_model(self, model_uri: str, name: str) -> Optional[str]:
        """
        Enregistre un modèle dans le registry.
        
        Args:
            model_uri: URI du modèle (runs:/xxx/model)
            name: Nom du modèle dans le registry
            
        Returns:
            Version du modèle ou None
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            return None
        
        try:
            result = mlflow.register_model(model_uri, name)
            return result.version
        except Exception as e:
            print(f"⚠️ Erreur registration model: {e}")
            return None
    
    def get_model_versions(self, name: str) -> List[ModelVersion]:
        """
        Liste les versions d'un modèle.
        
        Args:
            name: Nom du modèle
            
        Returns:
            Liste de ModelVersion
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            return []
        
        try:
            versions = self.client.search_model_versions(f"name='{name}'")
            
            result = []
            for v in versions:
                # Récupérer les métriques du run
                accuracy = 0.0
                try:
                    run = self.client.get_run(v.run_id)
                    accuracy = run.data.metrics.get('accuracy', 0.0)
                except:
                    pass
                
                result.append(ModelVersion(
                    name=v.name,
                    version=v.version,
                    run_id=v.run_id,
                    stage=v.current_stage,
                    accuracy=accuracy,
                    created_at=str(v.creation_timestamp)
                ))
            
            return result
            
        except Exception as e:
            print(f"⚠️ Erreur list versions: {e}")
            return []
    
    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """
        Récupère le modèle en production.
        
        Args:
            name: Nom du modèle
            
        Returns:
            ModelVersion ou None
        """
        versions = self.get_model_versions(name)
        
        for v in versions:
            if v.stage == "Production":
                return v
        
        return None
    
    def get_staging_model(self, name: str) -> Optional[ModelVersion]:
        """
        Récupère le modèle en staging.
        
        Args:
            name: Nom du modèle
            
        Returns:
            ModelVersion ou None
        """
        versions = self.get_model_versions(name)
        
        for v in versions:
            if v.stage == "Staging":
                return v
        
        return None
    
    def promote_to_staging(self, name: str, version: str) -> Tuple[bool, str]:
        """
        Promeut une version vers Staging.
        
        Args:
            name: Nom du modèle
            version: Version à promouvoir
            
        Returns:
            (success, message)
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            return False, "MLflow non disponible"
        
        try:
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage="Staging"
            )
            return True, f"✅ {name} v{version} → Staging"
        except Exception as e:
            return False, f"❌ Erreur: {e}"
    
    def promote_to_production(self, name: str, version: str,
                              archive_current: bool = True) -> Tuple[bool, str]:
        """
        Promeut une version vers Production.
        
        Args:
            name: Nom du modèle
            version: Version à promouvoir
            archive_current: Archiver le modèle production actuel
            
        Returns:
            (success, message)
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            return False, "MLflow non disponible"
        
        try:
            # Archiver le modèle actuel si demandé
            if archive_current:
                current = self.get_production_model(name)
                if current:
                    self.client.transition_model_version_stage(
                        name=name,
                        version=current.version,
                        stage="Archived"
                    )
            
            # Promouvoir le nouveau
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage="Production"
            )
            
            return True, f"✅ {name} v{version} → Production"
            
        except Exception as e:
            return False, f"❌ Erreur: {e}"
    
    def load_production_model(self, name: str):
        """
        Charge le modèle en production.
        
        Args:
            name: Nom du modèle
            
        Returns:
            Modèle PyTorch ou None
        """
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            model_uri = f"models:/{name}/Production"
            model = mlflow.pytorch.load_model(model_uri)
            return model
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle production: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SEARCH & COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_best_run(self, experiment_name: str, 
                     metric: str = "accuracy",
                     ascending: bool = False) -> Optional[Dict]:
        """
        Récupère le meilleur run d'une expérience.
        
        Args:
            experiment_name: Nom de l'expérience
            metric: Métrique à optimiser
            ascending: True si on minimise
            
        Returns:
            Dict avec infos du run ou None
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            return None
        
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                return None
            
            order = "ASC" if ascending else "DESC"
            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=[f"metrics.{metric} {order}"],
                max_results=1
            )
            
            if runs:
                run = runs[0]
                return {
                    'run_id': run.info.run_id,
                    'metrics': dict(run.data.metrics),
                    'params': dict(run.data.params),
                    'status': run.info.status
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Erreur recherche best run: {e}")
            return None
    
    def compare_runs(self, run_ids: List[str]) -> List[Dict]:
        """
        Compare plusieurs runs.
        
        Args:
            run_ids: Liste des run IDs
            
        Returns:
            Liste de dicts avec métriques
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            return []
        
        results = []
        
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                results.append({
                    'run_id': run_id,
                    'status': run.info.status,
                    'metrics': dict(run.data.metrics),
                    'params': dict(run.data.params)
                })
            except Exception as e:
                print(f"⚠️ Erreur run {run_id}: {e}")
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UTILS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_tracking_uri(self) -> str:
        """Retourne l'URI de tracking."""
        return self.mlflow_config.tracking_uri
    
    def get_experiment_url(self, experiment_name: str) -> Optional[str]:
        """Retourne l'URL de l'expérience dans l'UI MLflow."""
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp:
                base_url = self.mlflow_config.tracking_uri.replace("/api", "")
                return f"{base_url}/#/experiments/{exp.experiment_id}"
        except:
            pass
        
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_mlflow_instance: Optional[MLflowManager] = None

def get_mlflow_manager() -> MLflowManager:
    """Singleton pour obtenir le manager MLflow."""
    global _mlflow_instance
    if _mlflow_instance is None:
        _mlflow_instance = MLflowManager()
    return _mlflow_instance


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Test MLflowManager...")
    
    manager = get_mlflow_manager()
    
    if MLFLOW_AVAILABLE:
        print(f"Tracking URI: {manager.get_tracking_uri()}")
        
        # Test start run
        run_id = manager.start_run(
            MLflowManager.EXPERIMENT_BRAND_CLASSIFICATION,
            run_name="test-run"
        )
        print(f"Run ID: {run_id}")
        
        if run_id:
            # Log params
            manager.log_params({
                'model': 'efficientnet_b4',
                'batch_size': 32,
                'learning_rate': 0.001
            })
            
            # Log metrics
            manager.log_metrics({
                'accuracy': 0.85,
                'loss': 0.32
            })
            
            manager.end_run()
            print("✅ Run terminé")
        
        # List versions
        versions = manager.get_model_versions(MLflowManager.MODEL_EFFICIENTNET)
        print(f"Versions {MLflowManager.MODEL_EFFICIENTNET}: {len(versions)}")
        
    else:
        print("⚠️ MLflow non installé - test limité")
    
    print("✅ Tests OK")
