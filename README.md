# 🚗 ALPR — Contrôle d'accès par lecture de plaque

Contrôle l'accès d'un parking privé par lecture de plaque, détecte les plaques usurpées en comparant la marque observée à la marque déclarée, et réentraîne automatiquement son modèle de marque à partir des corrections de l'opérateur.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![MLflow](https://img.shields.io/badge/MLflow-3.1.1-green)
![Airflow](https://img.shields.io/badge/Airflow-2.7-teal)

```bash
git clone -b dev_pieric https://github.com/pf6511/alpr-mlops-platform
```

## 🎯 Fonctionnalités

| Domaine | Fonctionnalité |
|---|---|
| Inférence | 4 modèles en deux branches : YOLO (plaque) → fast-plate-ocr (texte) ; YOLOv8s COCO (véhicule) → EfficientNet B0 (marque, 22 classes) |
| Métier | Décision d'accès (résident connu et accès autorisé), alerte quand la marque prédite ≠ marque déclarée |
| API | API REST sécurisée (clé d'API) pour les équipements du parking : image → décision d'accès motivée ; l'interface Gradio reste le back office |
| Apprentissage actif | Image de l'incohérence conservée, validée par l'opérateur, ajoutée au jeu d'entraînement `labeled/<marque>/` |
| MLOps | Registre MLflow (alias `champion` / `challenger`), paquets de modèle sans code, manifeste de lignage, règle de promotion explicite |
| Réentraînement automatique | Cycle déclenché par les données (≥ 20 nouvelles images validées), orchestré par Airflow ; l'application charge le nouveau champion sans redémarrer |
| Monitoring | Chaque passage est journalisé avec ses mesures ; rapport quotidien (Airflow) : indicateurs, seuils du cahier des charges, dérive des données (Evidently) ; alerte si un seuil est dépassé |
| RGPD | Image conservée seulement en cas d'incohérence, plaque masquée et nom aléatoire, supprimée à 30 jours par le stockage (cycle de vie S3) ; jeu d'entraînement anonymisé (aucun lien avec le passage) |
| CI/CD | GitHub Actions : tests à chaque push et pull request, puis publication de l'image sur GHCR |

## 🏗️ Architecture

```
POINTS D'ENTRÉE   Interface Gradio (src/ui, back office) · API REST (src/api, bornes) · CLI · Ordonnanceurs (DAG Airflow, alpr-retrain)
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
│   ├── api/                   # API REST FastAPI des équipements
│   ├── data/                  # Base, stockage S3, rétention RGPD
│   ├── models/                # Détecteur d'incohérences, façade historique
│   ├── mlops/                 # Registre, bootstrap, promotion, cycle automatique
│   ├── monitoring/            # Rapport de monitoring (indicateurs, seuils, Evidently)
│   ├── training/              # Jeu de données, entraînement, métriques
│   └── serving/               # API Ray Serve (option de montée en charge)
├── pipelines/                 # CLI d'inférence
├── airflow/dags/              # DAG alpr_brand_retraining et alpr_monitoring
├── docker/                    # Dockerfile, docker-compose.yml, surcharge de production
├── migrations/                # Schéma SQL
├── models/                    # Poids locaux (détecteur de plaques, repli du modèle de marque)
└── tests/                     # Tests unitaires et d'intégration
```

## 🚀 Démarrage

```bash
cd docker
cp env.cloud.example .env        # puis compléter (MLflow, AWS)
docker compose up -d             # application, PostgreSQL, MinIO
```

| Service | Adresse | Démarrage |
|---|---|---|
| Application | http://localhost:7860 | `docker compose up -d` |
| MinIO | http://localhost:9001 | `docker compose up -d` |
| API REST (documentation interactive) | http://127.0.0.1:8000/docs | `docker compose --profile api up -d alpr-api` (exige `ALPR_API_KEY`) |
| Airflow (`admin` / `AIRFLOW_PASSWORD`, `admin` par défaut) | http://localhost:8080 | `docker compose --profile airflow up -d airflow` |
| Réentraînement planifié (sans Airflow) | — | `docker compose --profile mlops up -d alpr-retrain` |

**Première installation** (une fois la stack démarrée) :

```bash
docker compose exec alpr-engine python -m src.data.retention --apply   # suppression des captures à 30 jours (RGPD)
docker compose exec alpr-engine python -m src.data.retention           # vérifie la règle en place
docker compose exec alpr-engine python -m src.mlops.bootstrap          # si MODEL_BRAND_SOURCE=registry : v1 champion
```

La règle de conservation est portée par le bucket MinIO : à reposer après toute recréation du stockage (`down -v`).

**Production** : l'image publiée par la CI/CD remplace le build local et le code n'est plus monté.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 🧰 Commandes essentielles

Depuis le dossier `docker/`. Les services `alpr-api`, `alpr-retrain` et `airflow` ont un profil : les commandes globales doivent nommer les profils.

**Conteneurs**

| Commande | Effet |
|---|---|
| `docker compose up -d alpr-engine` | Démarre l'application (PostgreSQL démarre d'abord) |
| `docker compose --profile api up -d alpr-api` | Démarre l'API REST |
| `docker compose --profile airflow up -d airflow` | Démarre Airflow |
| `docker compose ps` | État des conteneurs (`healthy` après environ 1 min) |
| `docker compose logs --tail 50 alpr-engine` | Derniers journaux d'un service |
| `docker compose restart alpr-engine` | Recharge le code modifié (monté en lecture seule) |
| `docker compose --profile full --profile mlops --profile airflow stop` | Arrête tout |
| `docker compose --profile full --profile mlops --profile airflow down -v` | ⚠️ Supprime conteneurs **et données** (base, MinIO, caches) |

**Tests et qualité** (à la racine du dépôt)

```bash
# Suite complète dans l'image (root : écriture des fichiers temporaires dans le code monté)
docker run --rm -u root -v "${PWD}:/app" -w /app alpr-engine:local python -m pytest tests -q

# Un seul test, trace courte
docker run --rm -u root -v "${PWD}:/app" -w /app alpr-engine:local python -m pytest tests/test_privacy.py -q --tb=short

# Analyse statique (noms non définis, imports inutilisés)
ruff check .github tests airflow
```

**Fichiers dans le conteneur**

```bash
docker cp alpr-engine:/app/data/monitoring ./monitoring    # récupérer les rapports
docker compose exec -u root alpr-engine rm -rf /app/data/base   # supprimer un dossier copié par docker cp (propriétaire root)
```

Sous Windows, utiliser `127.0.0.1` plutôt que `localhost` pour joindre les ports publiés.

## 🔌 API REST

Point d'entrée des équipements (caméra, barrière) : elle traite un passage avec les mêmes services que l'interface (même moteur, même champion, même base). La gestion (résidents, validation, historique) reste dans le back office Gradio.

| Route | Clé | Entrée | Sortie |
|---|---|---|---|
| `GET /health` | non | — | état du service, version du modèle de marque |
| `POST /v1/access` | `X-API-Key` | `image` (JPEG ou PNG, 5 Mo max), `gate_id` (facultatif) | plaque, décision (`authorized`, `reason` : `AUTHORIZED`, `UNKNOWN_PLATE`, `ACCESS_REVOKED`, `NO_PLATE`), marque, incohérence, version du modèle |

Sécurité : clé obligatoire (l'API refuse de démarrer sans `ALPR_API_KEY`), port publié sur `127.0.0.1` seulement, taille et format d'image contrôlés, erreurs sans détail interne, réponse minimale (ni identité du résident, ni image). Chaque réponse porte un `request_id` (aussi dans l'en-tête `X-Request-ID`) que l'on retrouve dans les logs.

```bash
curl -H "X-API-Key: $ALPR_API_KEY" -F image=@voiture.jpg -F gate_id=entree-nord http://127.0.0.1:8000/v1/access
```

L'API écoute sur `127.0.0.1` uniquement : sous Windows, utiliser `127.0.0.1` plutôt que `localhost` (qui peut désigner l'adresse IPv6 `::1`).

## 📈 Monitoring

Chaque passage (interface ou API) est journalisé avec ses mesures : motif de la décision, confiances, luminosité et netteté de l'image, temps de traitement, source, version du modèle. Le monitoring lit la vue `monitoring.passages` (le journal **sans les plaques**) et compare les 7 derniers jours aux 30 précédents.

| Niveau | Indicateurs | Seuil (`config.yaml`, section `monitoring`) |
|---|---|---|
| Service | passages, part des refus, temps de traitement (p95) | p95 ≤ 3 000 ms |
| Données | luminosité, netteté, distribution des marques prédites ; dérive (Evidently, `DataDriftPreset`) | moins de 50 % des colonnes en dérive |
| Modèle | confiances de la marque et de l'OCR, part des incohérences, **taux d'erreur réel** (validations de l'opérateur) | erreur ≤ 20 %, confiance ≥ 0,70 |

Le DAG `alpr_monitoring` (quotidien) exécute `monitoring_report` puis `check_alerts`, qui échoue (rouge) si un seuil est dépassé. Chaque exécution ajoute une ligne à `data/monitoring/history.jsonl` et, avec assez de passages, produit un rapport HTML Evidently :

```bash
docker exec -w /app alpr-engine python -m src.monitoring.report      # rapport à la demande
docker exec alpr-engine ls data/monitoring                            # rapports produits
docker cp alpr-engine:/app/data/monitoring/<rapport>.html .           # le copier pour l'ouvrir
```

Démonstration : `python -m src.monitoring.demo_data` insère des passages fictifs (`source = 'demo'`) qui simulent une dégradation (images sombres, confiance en baisse, erreurs) ; `--delete` les supprime.

Limite : le taux d'erreur ne porte que sur les incohérences validées par l'opérateur (indicateur partiel).

## 🔁 Commandes MLOps

À lancer dans le conteneur : `docker exec -w /app alpr-engine <commande>`.

| Commande | Effet |
|---|---|
| `python -m src.mlops.bootstrap` | Modèle local → version 1, champion (une fois) |
| `python -m src.training.train --epochs 3 [--base-dir <dossier>/<marque>/]` | Entraînement → nouvelle version challenger |
| `python -m src.mlops.promote` | Challenger contre champion sur le même jeu de test |
| `python -m src.mlops.continuous [--check \| --force \| --loop]` | Cycle complet : décision, entraînement, promotion |
| `python -m src.data.retention [--apply]` | Affiche ou applique la conservation de 30 jours |
| `python -m src.monitoring.report [--no-drift]` | Rapport de monitoring : indicateurs, seuils, dérive (Evidently) |
| `python -m src.monitoring.demo_data [--delete]` | Insère ou supprime les passages de démonstration |
| `python -m pipelines.inference_pipeline <image>` | Inférence en ligne de commande |

Le cycle automatique s'entraîne sur les seules images validées (`labeled/<marque>/`). Pour y ajouter le **socle de référence** (jeu public VMMRdb Europe, dont les 22 dossiers sont les 22 classes du modèle), l'entraînement se lance à la main :

```powershell
# 1. Socle reproductible : les 15 premières images de chaque marque (Windows, PowerShell)
powershell -ExecutionPolicy Bypass -File .\scripts\build_socle.ps1 -Source "C:\...\VMMRdb_europe" -Destination "C:\...\socle"

# 2. Copie dans le conteneur, entraînement, promotion
docker compose exec -u root alpr-engine rm -rf /app/data/base
docker cp C:\...\socle alpr-engine:/app/data/base
docker compose exec alpr-engine python -m src.training.train --base-dir data/base
docker compose exec alpr-engine python -m src.mlops.promote
```

Lancés à la main, `train` et `promote` ne prennent pas le verrou du cycle : arrêter `alpr-retrain` au préalable. Le manifeste de chaque version liste les images utilisées (`counts.by_origin`, `test_set`) et les marques écartées (`rejected_unknown_labels`).

## ⚙️ Configuration

Les valeurs sont dans `configs/config.yaml` (seuils, buckets, `retention`, `continuous_training`, `monitoring`, `mlflow`) et se surchargent par variables d'environnement. Principales variables (`docker/env.cloud.example`) :

| Variable | Rôle |
|---|---|
| `DATABASE_URL` | Base PostgreSQL de l'application |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_TOKEN` | Registre MLflow (Space privé) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | Artefacts MLflow sur S3 |
| `MODEL_BRAND_SOURCE` | `registry` = servir le champion du registre (`local` par défaut) |
| `S3_ENDPOINT` | MinIO par défaut ; vide = AWS S3 |
| `RETRAIN_MIN_NEW_IMAGES`, `RETRAIN_INTERVAL_HOURS` | Réentraînement automatique |
| `ALPR_API_KEY`, `ALPR_API_MAX_IMAGE_MB` | Clé de l'API REST (obligatoire), taille maximale d'une image |

## 🧪 Tests et CI/CD

```bash
python -m pytest tests -q        # environnement Python local ; dans Docker : voir « Commandes essentielles »
```

`.github/workflows/ci.yml` : à chaque push et pull request, lint, build des images, démarrage de la stack, test HTTP de l'interface, démarrage de l'API (santé, refus sans clé) et suite de tests (dont le cycle MLOps complet avec un vrai MLflow). Après une CI verte sur `dev_pieric`, `dev` ou `main`, le job `publish-image` publie `ghcr.io/pf6511/alpr-engine` (étiquettes de branche et `sha-<commit>`).

## 📊 Modèles

| Modèle | Rôle | Source |
|---|---|---|
| YOLO (`best_model_detection_plaque.pt`) | Trouver la plaque | Fichier du dépôt |
| fast-plate-ocr (`global-plates-mobile-vit-v2-model`) | Lire la plaque | Pré-entraîné, téléchargé |
| YOLOv8s COCO | Trouver le véhicule | Pré-entraîné, téléchargé |
| EfficientNet B0, 22 marques (classes du jeu VMMRdb Europe) | Reconnaître la marque | Registre MLflow (champion), repli `models/best_model_efficientNet_finetune.pth` |

## 📝 License

MIT License - ACT-IA © 2025

## 👤 Auteurs

**Mickael Moisan** - ACT-IA
- GitHub: [@actia-mickael](https://github.com/actia-mickael)
- Email: contact@act-ia.fr
