# 🚗 ALPR Engine - LEAD Edition

**Github
Git clone https://github.com/pf6511/alpr-mlops-platform


**Automatic License Plate Recognition** avec classification de marques et MLOps pipeline.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.9+-green)

## 🎯 Fonctionnalités

- **Détection de plaques** : YOLOv8 fine-tuné
- **OCR** : fast-plate-ocr (40+ pays EU)
- **Détection véhicule** : YOLOv8s (COCO)
- **Classification marque** : EfficientNet B4 (22 marques EU)
- **Mismatch Detection** : Détection automatique des erreurs de classification
- **MLOps Pipeline** : Retraining automatique via Airflow + MLflow

## 📁 Structure du projet

```
alpr-mlops-platform/
├── src/
│   ├── data/              # Database, Storage S3
│   ├── features/          # Feature engineering (futur)
│   ├── models/            # Pipeline, Mismatch detector, MLflow client
│   ├── utils/             # UI components, visualizer
│   └── serving/           # Ray Serve deployment
├── configs/
│   ├── config.yaml        # Configuration centralisée
│   └── settings.py        # Chargement config + env vars
├── pipelines/
│   ├── training_pipeline.py
│   └── inference_pipeline.py
├── airflow/
│   └── dags/              # DAGs Airflow
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── k8s/                   # Kubernetes manifests
├── helm/
│   └── alpr-chart/        # Helm chart
├── models/                # Poids des modèles
├── migrations/            # Scripts SQL
├── tests/
├── notebooks/
├── docs/
├── app.py                 # Application Gradio
└── requirements.txt
```

## 🚀 Quick Start

### Installation locale

```bash
# Clone
git clone https://github.com/actia-mickael/alpr-engine.git
cd alpr-engine

# Environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Dependencies
pip install -r requirements.txt

# Run
python app.py
```

### Docker

```bash
cd docker
docker compose up -d

# URLs:
# - App: http://localhost:7860
# - MLflow: http://localhost:5000
# - MinIO: http://localhost:9001
```

### Docker avec Airflow

```bash
docker compose --profile full up -d

# Airflow: http://localhost:8080 (admin/admin)
```

## ⚙️ Configuration

### Variables d'environnement

```bash
# Copier le template
cp .env.example .env

# Éditer selon votre environnement
```

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment (development/production) | development |
| `DB_MODE` | sqlite ou postgres | sqlite |
| `DB_HOST` | Host PostgreSQL | localhost |
| `STORAGE_MODE` | local ou s3 | local |
| `S3_ENDPOINT` | Endpoint S3/MinIO | - |
| `MLFLOW_TRACKING_URI` | URI MLflow | ./mlruns |
| `RAY_MODE` | local ou serve | local |

### config.yaml

```yaml
env: development

database:
  mode: sqlite
  sqlite_path: ./data/alpr.db

storage:
  mode: local
  local_path: ./data/storage

models:
  thresholds:
    detection: 0.5
    ocr: 0.6
    brand: 0.7
```

## 🔧 Pipelines

### Inference

```bash
# CLI
python -m pipelines.inference_pipeline image.jpg --declared-brand Renault

# Python
from pipelines import InferencePipeline

pipeline = InferencePipeline()
result = pipeline.process("image.jpg", run_brand=True)
```

### Training

```bash
# CLI
python -m pipelines.training_pipeline --source ./data/labeled

# Python
from pipelines import TrainingPipeline

pipeline = TrainingPipeline()
result = pipeline.run()
```

## 🧪 Tests

```bash
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_basic.py -v
```

## 📊 Modèles

| Modèle | Fichier | Description |
|--------|---------|-------------|
| YOLO Plaque | `best_model_detection_plaque.pt` | YOLOv8 fine-tuné |
| YOLO Véhicule | `yolov8s.pt` | YOLOv8s COCO |
| EfficientNet | `best_model_efficientNet_finetune.pth` | 22 marques EU |
| OCR | `global-plates-mobile-vit-v2-model` | Auto-download |

## 🏗️ Architecture MLOps

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Gradio    │────▶│   Pipeline  │────▶│  Mismatch   │
│     App     │     │  Inference  │     │  Detector   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Airflow   │────▶│  Training   │────▶│   MLflow    │
│    DAGs     │     │  Pipeline   │     │  Registry   │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 📝 License

MIT License - ACT-IA © 2025

## 👤 Auteur

**Mickael Moisan** - ACT-IA
- GitHub: [@actia-mickael](https://github.com/actia-mickael)
- LinkedIn: [mickael-moisan-42605774](https://linkedin.com/in/mickael-moisan-42605774)
- Email: contact@act-ia.fr
