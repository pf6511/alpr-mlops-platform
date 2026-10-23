"""
ALPR Engine - Settings (LEAD Edition)
Charge la configuration depuis config.yaml et les variables d'environnement.
"""

import os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Charger .env si présent
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Charger YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatabaseSettings:
    """Configuration base de données."""
    mode: str = "sqlite"
    sqlite_path: str = "./data/alpr.db"
    host: str = "localhost"
    port: int = 5432
    name: str = "alpr"
    user: str = "alpr"
    password: str = ""
    sslmode: str = "prefer" 


@dataclass
class S3Settings:
    """Configuration stockage S3."""
    mode: str = "local"
    local_path: str = "./data/storage"
    endpoint: str = ""
    access_key: str = ""
    secret_key: str = ""
    region: str = "us-east-1"
    bucket_captures: str = "alpr-captures"
    bucket_dataset: str = "alpr-dataset"
    # Bucket des artefacts (modèles) : celui de MLflow dans docker-compose
    artifacts_bucket: str = "mlflow"


@dataclass
class MLflowSettings:
    """
    Configuration MLflow (suivi des expériences et registre de modèles).

    Le jeton d'un serveur protégé se passe par la variable d'environnement
    MLFLOW_TRACKING_TOKEN, lue directement par MLflow (jamais dans ce fichier).
    """
    tracking_uri: str = "./mlruns"
    experiment_prefix: str = "alpr"
    experiment: str = "alpr-brand-classifier"
    brand_model_name: str = "vehicle_brand_model"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"
    # Origine du classificateur de marque : "registry" (MLflow, repli local) ou "local"
    brand_model_source: str = "local"
    cache_dir: str = "data/registry_cache"


@dataclass
class RayServeSettings:
    """Configuration Ray Serve."""
    mode: str = "local"
    address: str = "auto"
    http_port: int = 8000


@dataclass
class AirflowSettings:
    """Configuration Airflow."""
    min_dataset_size: int = 500
    retrain_schedule: str = "@daily"


@dataclass
class ModelSettings:
    """Configuration des modèles."""
    yolo_plate_path: str = "models/best_model_detection_plaque.pt"
    yolo_vehicle_path: str = "models/yolov8s.pt"
    efficientnet_path: str = "models/best_model_efficientNet_finetune.pth"
    brand_classes: List[str] = field(default_factory=lambda: [
        "audi", "bmw", "citroen", "dacia", "fiat", "ford", "honda",
        "hyundai", "kia", "mazda", "mercedes", "nissan", "opel",
        "peugeot", "renault", "seat", "skoda", "suzuki", "toyota",
        "volkswagen", "volvo", "other"
    ])
    ocr_model: str = "global-plates-mobile-vit-v2-model"
    detection_threshold: float = 0.25
    vehicle_threshold: float = 0.25
    ocr_threshold: float = 0.6
    brand_threshold: float = 0.7


@dataclass
class RetentionSettings:
    """
    Durées de conservation des données personnelles (RGPD, art. 5.1.e).

    Les images d'incohérences (véhicule et plaque visibles) ne servent qu'à la
    validation par un opérateur : au-delà de `captures_days`, le stockage les
    supprime lui-même (règle de cycle de vie S3, voir src/data/retention.py).
    """
    captures_days: int = 30
    captures_prefix: str = "mismatches/"


@dataclass
class GradioSettings:
    """Configuration Gradio."""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    debug: bool = False


@dataclass
class Settings:
    """Configuration globale."""
    env: str = "development"
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    s3: S3Settings = field(default_factory=S3Settings)
    mlflow: MLflowSettings = field(default_factory=MLflowSettings)
    ray: RayServeSettings = field(default_factory=RayServeSettings)
    airflow: AirflowSettings = field(default_factory=AirflowSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    gradio: GradioSettings = field(default_factory=GradioSettings)
    retention: RetentionSettings = field(default_factory=RetentionSettings)


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_env_vars(value: str) -> str:
    """Résout les variables d'environnement dans une string ${VAR:default}."""
    if not isinstance(value, str):
        return value
    
    import re
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
    
    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)
    
    return re.sub(pattern, replacer, value)


def _load_yaml_config() -> Dict:
    """Charge le fichier config.yaml."""
    if not YAML_AVAILABLE:
        return {}
    
    config_paths = [
        Path(__file__).parent / "config.yaml",
        Path("configs/config.yaml"),
        Path("config.yaml"),
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Résoudre les variables d'environnement
            def resolve_recursive(obj):
                if isinstance(obj, dict):
                    return {k: resolve_recursive(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [resolve_recursive(v) for v in obj]
                elif isinstance(obj, str):
                    return _resolve_env_vars(obj)
                return obj
            
            return resolve_recursive(config)
    
    return {}


def _create_settings_from_config(config: Dict) -> Settings:
    """Crée un objet Settings depuis la config YAML."""
    
    # Database
    db_config = config.get('database', {})
    pg_config = db_config.get('postgres', {})
    database = DatabaseSettings(
        mode=os.environ.get('DB_MODE', db_config.get('mode', 'sqlite')),
        sqlite_path=db_config.get('sqlite_path', './data/alpr.db'),
        host=os.environ.get('DB_HOST', pg_config.get('host', 'localhost')),
        port=int(os.environ.get('DB_PORT', pg_config.get('port', 5432))),
        name=os.environ.get('DB_NAME', pg_config.get('name', 'alpr')),
        user=os.environ.get('DB_USER', pg_config.get('user', 'alpr')),
        password=os.environ.get('DB_PASSWORD', pg_config.get('password', '')),
        sslmode=os.environ.get('DB_SSLMODE', pg_config.get('sslmode', 'prefer')),
    )
    
    # S3
    s3_config = config.get('storage', {})
    s3_details = s3_config.get('s3', {})
    buckets = s3_details.get('buckets', {})
    s3 = S3Settings(
        mode=os.environ.get('STORAGE_MODE', s3_config.get('mode', 'local')),
        local_path=s3_config.get('local_path', './data/storage'),
        endpoint=os.environ.get('S3_ENDPOINT', s3_details.get('endpoint', '')),
        access_key=os.environ.get('S3_ACCESS_KEY', s3_details.get('access_key', '')),
        secret_key=os.environ.get('S3_SECRET_KEY', s3_details.get('secret_key', '')),
        region=os.environ.get('S3_REGION', s3_details.get('region', 'us-east-1')),
        bucket_captures=os.environ.get('S3_BUCKET_CAPTURES', buckets.get('captures', 'alpr-captures')),
        bucket_dataset=os.environ.get('S3_BUCKET_DATASET', buckets.get('dataset', 'alpr-dataset')),
        artifacts_bucket=os.environ.get('S3_BUCKET_ARTIFACTS', buckets.get('artifacts', 'mlflow')),
    )
    
    # MLflow
    mlflow_config = config.get('mlflow', {})
    mlflow = MLflowSettings(
        tracking_uri=os.environ.get('MLFLOW_TRACKING_URI', mlflow_config.get('tracking_uri', './mlruns')),
        experiment_prefix=mlflow_config.get('experiment_prefix', 'alpr'),
        experiment=os.environ.get('MLFLOW_EXPERIMENT', mlflow_config.get('experiment', 'alpr-brand-classifier')),
        brand_model_name=os.environ.get('MLFLOW_BRAND_MODEL', mlflow_config.get('brand_model_name', 'vehicle_brand_model')),
        champion_alias=mlflow_config.get('champion_alias', 'champion'),
        challenger_alias=mlflow_config.get('challenger_alias', 'challenger'),
        brand_model_source=os.environ.get('MODEL_BRAND_SOURCE', mlflow_config.get('brand_model_source', 'local')),
        cache_dir=mlflow_config.get('cache_dir', 'data/registry_cache'),
    )
    
    # Ray
    ray_config = config.get('ray', {})
    ray = RayServeSettings(
        mode=os.environ.get('RAY_MODE', ray_config.get('mode', 'local')),
        address=ray_config.get('address', 'auto'),
        http_port=int(os.environ.get('RAY_HTTP_PORT', ray_config.get('http_port', 8000))),
    )
    
    # Airflow
    airflow_config = config.get('airflow', {})
    airflow = AirflowSettings(
        min_dataset_size=int(airflow_config.get('min_dataset_size', 500)),
        retrain_schedule=airflow_config.get('retrain_schedule', '@daily'),
    )
    
    # Models
    models_config = config.get('models', {})
    paths = models_config.get('paths', {})
    thresholds = models_config.get('thresholds', {})
    models = ModelSettings(
        yolo_plate_path=paths.get('yolo_plate', 'models/best_model_detection_plaque.pt'),
        yolo_vehicle_path=paths.get('yolo_vehicle', 'models/yolov8s.pt'),
        efficientnet_path=paths.get('efficientnet', 'models/best_model_efficientNet_finetune.pth'),
        brand_classes=models_config.get('brand_classes', ModelSettings().brand_classes),
        ocr_model=os.environ.get('MODEL_OCR_NAME', models_config.get('ocr_model', ModelSettings().ocr_model)),
        detection_threshold=float(os.environ.get('MODEL_YOLO_CONF', thresholds.get('detection', 0.25))),
        vehicle_threshold=float(thresholds.get('vehicle', 0.25)),
        ocr_threshold=float(thresholds.get('ocr', 0.6)),
        brand_threshold=float(thresholds.get('brand', 0.7)),
    )
    
    # Gradio
    gradio_config = config.get('gradio', {})
    gradio = GradioSettings(
        server_name=os.environ.get('GRADIO_SERVER_NAME', gradio_config.get('server_name', '0.0.0.0')),
        server_port=int(os.environ.get('GRADIO_SERVER_PORT', gradio_config.get('server_port', 7860))),
        share=gradio_config.get('share', False),
        debug=os.environ.get('GRADIO_DEBUG', str(gradio_config.get('debug', False))).lower() == 'true',
    )
    
    # Conservation des données (RGPD)
    retention_config = config.get('retention', {})
    retention = RetentionSettings(
        captures_days=int(os.environ.get('CAPTURES_RETENTION_DAYS', retention_config.get('captures_days', 30))),
        captures_prefix=retention_config.get('captures_prefix', 'mismatches/'),
    )

    return Settings(
        env=os.environ.get('ENV', config.get('env', 'development')),
        database=database,
        s3=s3,
        mlflow=mlflow,
        ray=ray,
        airflow=airflow,
        models=models,
        gradio=gradio,
        retention=retention,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retourne la configuration (singleton)."""
    config = _load_yaml_config()
    return _create_settings_from_config(config)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    settings = get_settings()
    print(f"Environment: {settings.env}")
    print(f"Database mode: {settings.database.mode}")
    print(f"Storage mode: {settings.s3.mode}")
    print(f"MLflow URI: {settings.mlflow.tracking_uri}")
    print(f"Gradio port: {settings.gradio.server_port}")
    print(f"Brand classes: {len(settings.models.brand_classes)}")
