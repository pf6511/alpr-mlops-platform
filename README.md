# 🚗 ALPR — Contrôle d'accès par lecture de plaque

Contrôle l'accès d'un parking privé par lecture de plaque, détecte les plaques usurpées en comparant la marque observée à la marque déclarée, et réentraîne automatiquement son modèle de marque à partir des corrections de l'opérateur.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![MLflow](https://img.shields.io/badge/MLflow-3.1.1-green)
![Airflow](https://img.shields.io/badge/Airflow-2.7-teal)

```bash
git clone https://github.com/pf6511/alpr-mlops-platform
```

## 🎯 Fonctionnalités

| Domaine | Fonctionnalité |
|---|---|
| Inférence | 4 modèles en deux branches : YOLO (plaque) → fast-plate-ocr (texte) ; YOLOv8s COCO (véhicule) → EfficientNet B0 (marque, 22 classes) |
| Métier | Décision d'accès (résident connu et accès autorisé), alerte quand la marque prédite ≠ marque déclarée |
| Apprentissage actif | Image de l'incohérence conservée, validée par l'opérateur, ajoutée au jeu d'entraînement `labeled/<marque>/` |
| MLOps | Registre MLflow (alias `champion` / `challenger`), paquets de modèle sans code, manifeste de lignage, règle de promotion explicite |
| Réentraînement automatique | Cycle déclenché par les données (≥ 20 nouvelles images validées), orchestré par Airflow ; l'application charge le nouveau champion sans redémarrer |
| RGPD | Image conservée seulement en cas d'incohérence, supprimée à 30 jours par le stockage (cycle de vie S3) |
| CI/CD | GitHub Actions : tests à chaque push et pull request, puis publication de l'image sur GHCR |

## 🏗️ Architecture

```
POINTS D'ENTRÉE   Interface Gradio (src/ui) · API Ray Serve · CLI · Ordonnanceurs (DAG Airflow, alpr-retrain)
        │
MÉTIER            src/services : capture (décision d'accès, incohérences), résidents, validation, système
        │         container.py : assemblage unique
CŒUR              src/inference (ALPREngine, contrats des modèles) · src/data · src/mlops + src/training
        │
STOCKAGE          PostgreSQL et MinIO (local, données personnelles) · MLflow : Space HF + Neon + S3 (cloud)
```

Chaque couche ne dépend que des couches inférieures ; seul `src/mlops/registry.py` parle à MLflow.

## 📁 Structure

```
alpr-mlops-platform/
├── app.py                     # Assemble les services et lance l'interface
├── configs/                   # config.yaml (valeurs) + settings.py (lecture + variables d'env)
├── src/
│   ├── inference/             # Moteur unique ALPREngine, modèles, chargement
│   ├── services/              # Règles métier, assemblage (container.py)
│   ├── ui/                    # Interface Gradio (un fichier par onglet)
│   ├── data/                  # Base, stockage S3, rétention RGPD
│   ├── models/                # Détecteur d'incohérences, façade historique
│   ├── mlops/                 # Registre, bootstrap, promotion, cycle automatique
│   ├── training/              # Jeu de données, entraînement, métriques
│   └── serving/               # API Ray Serve
├── pipelines/                 # CLI d'inférence
├── airflow/dags/              # DAG alpr_brand_retraining
├── docker/                    # Dockerfile, docker-compose.yml, surcharge de production
├── migrations/                # Schéma SQL
├── models/                    # Poids locaux (détecteur de plaques, repli du modèle de marque)
└── tests/                     # Tests unitaires et d'intégration
```

## 🚀 Démarrage

```bash
cd docker
cp env.cloud.example .env        # puis compléter (MLflow, AWS) — ne jamais versionner .env
docker compose up -d             # application, PostgreSQL, MinIO
```

| Service | Adresse | Démarrage |
|---|---|---|
| Application | http://localhost:7860 | `docker compose up -d` |
| MinIO | http://localhost:9001 | `docker compose up -d` |
| Airflow | http://localhost:8080 | `docker compose --profile airflow up -d airflow` |
| Réentraînement planifié (sans Airflow) | — | `docker compose --profile mlops up -d alpr-retrain` |

**Production** : l'image publiée par la CI/CD remplace le build local et le code n'est plus monté.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 🔁 Commandes MLOps

À lancer dans le conteneur : `docker exec -w /app alpr-engine <commande>`.

| Commande | Effet |
|---|---|
| `python -m src.mlops.bootstrap` | Modèle local → version 1, champion (une fois) |
| `python -m src.training.train --epochs 3 [--base-dir <dossier>/<marque>/]` | Entraînement → nouvelle version challenger |
| `python -m src.mlops.promote` | Challenger contre champion sur le même jeu de test |
| `python -m src.mlops.continuous [--check \| --force \| --loop]` | Cycle complet : décision, entraînement, promotion |
| `python -m src.data.retention [--apply]` | Affiche ou applique la conservation de 30 jours |
| `python -m pipelines.inference_pipeline <image>` | Inférence en ligne de commande |

## ⚙️ Configuration

Les valeurs sont dans `configs/config.yaml` (seuils, buckets, `retention`, `continuous_training`, `mlflow`) et se surchargent par variables d'environnement. Principales variables (`docker/env.cloud.example`) :

| Variable | Rôle |
|---|---|
| `DATABASE_URL` | Base PostgreSQL de l'application |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_TOKEN` | Registre MLflow (Space privé) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | Artefacts MLflow sur S3 |
| `MODEL_BRAND_SOURCE` | `registry` = servir le champion du registre (`local` par défaut) |
| `S3_ENDPOINT` | MinIO par défaut ; vide = AWS S3 |
| `RETRAIN_MIN_NEW_IMAGES`, `RETRAIN_INTERVAL_HOURS` | Réentraînement automatique |

## 🧪 Tests et CI/CD

```bash
python -m pytest tests -q
```

`.github/workflows/ci.yml` : à chaque push et pull request, lint, build des images, démarrage de la stack, test HTTP et suite de tests (dont le cycle MLOps complet avec un vrai MLflow). Après une CI verte sur `dev_pieric`, `dev` ou `main`, le job `publish-image` publie `ghcr.io/pf6511/alpr-engine` (étiquettes de branche et `sha-<commit>`).

## 📊 Modèles

| Modèle | Rôle | Source |
|---|---|---|
| YOLO (`best_model_detection_plaque.pt`) | Trouver la plaque | Fichier du dépôt |
| fast-plate-ocr (`global-plates-mobile-vit-v2-model`) | Lire la plaque | Pré-entraîné, téléchargé |
| YOLOv8s COCO | Trouver le véhicule | Pré-entraîné, téléchargé |
| EfficientNet B0, 22 marques | Reconnaître la marque | Registre MLflow (champion), repli `models/best_model_efficientNet_finetune.pth` |

## 📝 License

MIT License - ACT-IA © 2025

## 👤 Auteurs

**Mickael Moisan** - ACT-IA
- GitHub: [@actia-mickael](https://github.com/actia-mickael)
- Email: contact@act-ia.fr
