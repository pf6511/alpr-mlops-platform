# Documentation Technique Complète - Projet ALPR Parking

---

## Table des Matières

1.  [Introduction](#1-introduction)
2.  [Vue d'Ensembe (Overview)](#2-vue-densemble)
3.  [Guide Utilisateur](#3-guide-utilisateur)
    *   [Interface Web](#31-interface-web)
    *   [Administration & Base de Données](#32-administration--base-de-données)
4.  [Architecture Technique](#4-architecture-technique)
    *   [Pipeline de Traitement](#41-pipeline-de-traitement)
    *   [Modèles Personnalisés](#42-modèles-personnalisés)
    *   [Contrôle d'Accès](#43-contrôle-daccès)
    *   [Visualisation](#44-visualisation)
5.  [Nouvelles Fonctionnalités (v2)](#5-nouvelles-fonctionnalités-v2)
    *   [Vérification de Conformité](#51-vérification-de-conformité)
    *   [Apprentissage Actif (Active Learning)](#52-apprentissage-actif-active-learning)
6.  [Dépannage & Compatibilité](#6-dépannage--compatibilité)

---

## 1. Introduction

Bienvenue dans la documentation technique détaillée du projet **ALPR Parking**. Ce document regroupe l'ensemble des informations techniques, fonctionnelles et administratives du système de reconnaissance de plaques d'immatriculation.

L'objectif de ce projet est de fournir une solution transparente et pédagogique pour le contrôle d'accès automatisé, illustrant chaque étape du traitement IA : de l'image brute à la décision d'ouverture de barrière.

---

## 2. Vue d'Ensemble

Ce système est une application web interactive basée sur l'intelligence artificielle pour la gestion de parking.

### Structure du Projet (`demo/`)

```text
demo/
├── app.py                # Point d'entrée principal (Interface Gradio)
├── requirements.txt      # Dépendances Python
├── Dockerfile            # Configuration pour déploiement conteneurisé
├── deploy.sh             # Script d'automatisation du déploiement
├── models/               # Modèles IA (YOLOv8 .pt)
├── video/                # Vidéos de démonstration
├── outputs/              # Logs (access_log.csv) et vidéos traitées
└── utils/                # Logique métier modulaire
    ├── pipeline.py       # Coeur du système (YOLO + OCR)
    ├── access_control.py # Gestion de l'accès (Liste Blanche)
    ├── database.py       # Gestion Base de Données SQLite
    ├── video_processor.py # Traitement vidéo et GIF
    ├── visualizer.py     # Rendu graphique
    └── error_gallery.py  # Gestion des exemples d'erreurs
```

### Flux de Données

1.  **Entrée** : Image ou Vidéo téléchargée par l'utilisateur.
2.  **Détection** : YOLOv8 localise les plaques.
3.  **Extraction** : Découpage de la zone d'intérêt (ROI).
4.  **Lecture (OCR)** : `fast-plate-ocr` convertit l'image en texte.
5.  **Décision** : Comparaison avec la base de données résidents.
6.  **Sortie** : Affichage visuel (Cadre Vert/Rouge) et enregistrement dans les logs.

---

## 3. Guide Utilisateur

### 3.1 Interface Web

L'interface est construite avec **Gradio** et organisée en 4 onglets :

1.  **Image Processing** :
    *   Analyse d'images fixes.
    *   Visualisation étape par étape (Input -> Crop -> Result).
2.  **Video/GIF Processing** :
    *   **Mode Sample** : Rapide, extrait quelques images clés.
    *   **Mode Annotate** : Complet, génère une vidéo MP4 avec les plaques surlignées.
3.  **History** :
    *   Tableau des derniers accès (Logs).
4.  **Settings** :
    *   Configuration du Token Hugging Face pour l'export de données.
    *   Sélection du modèle IA à la volée.

### 3.2 Administration & Base de Données

Le système utilise une base de données **SQLite** locale (`demo/alpr.db`).

#### Gestion des Résidents (Onglet Administration)
*   **Recherche** : Filtrer par plaque, nom ou prénom.
*   **Ajout** : Formulaire pour inscrire un nouveau véhicule (Plaque unique requise).
*   **Modification** : Édition rapide en double-cliquant sur les cellules.
*   **Contrôle d'Accès** : Bouton "Toggle Accès" pour bloquer/débloquer un résident instantanément.
*   **Import CSV** : Si la base est vide, elle est automatiquement peuplée par `plaques_avec_donnees.csv`.

#### Structure des Données
*   **Table `residents`** : Liste des véhicules autorisés et infos propriétaires.
*   **Table `logs`** : Historique de tous les passages (Autorisé/Refusé).
*   **Table `inconsistencies`** : Journal des incohérences de marque (pour l'Active Learning).

---

## 4. Architecture Technique

### 4.1 Pipeline de Traitement (`utils/pipeline.py`)

La classe `ALPRPipeline` est le moteur du système :
1.  **Initialisation** : Charge YOLOv8 et OCR au démarrage.
2.  **`process_image`** : Exécute la chaîne de traitement complète.
3.  **Gestion Dynamique** : `reload_model()` permet de changer de modèle sans redémarrer le serveur, optimisant l'usage RAM.

### 4.2 Modèles Personnalisés (`models/`)

Le système supporte deux types de modèles YOLO :
1.  **Standard (Ultralytics)** : Modèles `.pt` classiques YOLOv8.
2.  **Custom ("From Scratch")** : Modèles à architecture personnalisée (ex: `modelemaison.pt`).
    *   Détectés automatiquement par leur nom de fichier.
    *   Utilisent une classe `SimpleYOLO` définie dans `utils/custom_yolo.py` pour recréer l'architecture PyTorch spécifique.

### 4.3 Contrôle d'Accès (`utils/access_control.py`)

*   **Logique** : Vérification stricte ou "floue" (Fuzzy Matching) contre la liste blanche.
*   **Normalisation** : Suppression des tirets/espaces pour la comparaison (ex: `AA-123-BB` = `AA123BB`).
*   **Fuzzy Matching** : Tolérance aux erreurs OCR mineures (ex: `8` vs `B`) si le score de similarité > 85%.

### 4.4 Visualisation (`utils/visualizer.py`)

Génère les overlays graphiques :
*   **Bannières** : "ACCÈS AUTORISÉ" (Vert) ou "ACCÈS REFUSÉ" (Rouge).
*   **Confiance** : Affichage des scores de probabilité pour la détection et l'OCR.
*   **Marque** : Affichage de la marque détectée vs attendue.

---

## 5. Nouvelles Fonctionnalités (v2)

### 5.1 Vérification de Conformité

Le système vérifie la cohérence entre le véhicule physique et la carte grise virtuelle :
*   **Marque Attendue** : Celle enregistrée dans la base (ex: Audi).
*   **Marque Détectée** : Celle reconnue par l'IA sur l'image (ex: BMW).
*   **Alerte** : Si différent -> "MARQUE INCORRECTE" (Orange).

### 5.2 Apprentissage Actif (Active Learning)

Transforme les erreurs en opportunités d'amélioration :
1.  **Détection** : Les incohérences sont stockées dans la table `inconsistencies`.
2.  **Correction** : L'administrateur corrige la vraie marque via l'interface.
3.  **Export** : L'image et le label corrigé sont envoyés vers un **Dataset Hugging Face** pour réentraînement futur.

Technologie : Utilise l'API `huggingface_hub` et un Token Write sécurisé.

---

## 6. Dépannage & Compatibilité

### Problèmes Courants

1.  **"Model not found"** :
    *   Vérifiez que le fichier `.pt` est bien dans `demo/models/`.
    *   Vérifiez le chemin dans `pipeline.py`.

2.  **Vidéos illisibles (Navigateur)** :
    *   Le système convertit automatiquement les sorties en **H.264** pour la compatibilité Web.
    *   Assurez-vous que `ffmpeg` est installé (inclus dans l'image Docker).

3.  **Erreur Base de Données ("Locked")** :
    *   Le timeout SQLite a été augmenté à 30s.
    *   Évitez d'ouvrir le fichier `.db` avec un logiciel externe pendant que l'app tourne.

### Performance
*   **CPU** : L'application est optimisée pour tourner sur CPU (Hugging Face Spaces Free Tier).
*   **RAM** : Consommation < 16Go grâce à la gestion dynamique des modèles.
*   **GPU** : Supporté nativement si CUDA est disponible (auto-détection).

---
*Document généré automatiquement pour le projet ALPR Parking.*
