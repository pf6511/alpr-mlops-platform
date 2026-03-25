# Documentation Technique - Application Web ALPR

Ce document détaille le fonctionnement interne de l'application web de reconnaissance de plaques d'immatriculation (ALPR).

---

## Structure du projet (`demo/`)

```text
demo/
├── app.py                # Point d'entrée principal (Interface Gradio)
├── requirements.txt      # Dépendances Python
├── Dockerfile            # Configuration pour Hugging Face Spaces
├── deploy.sh             # Script d'automatisation du déploiement
├── models/               # Modèles YOLOv8 (.pt)
├── video/                # Vidéos de démonstration
├── outputs/              # Logs (access_log.csv) et vidéos traitées
└── utils/                # Logique métier modulaire
    ├── pipeline.py       # Coeur du système (YOLO + OCR)
    ├── access_control.py # Gestion de l'accès parking
    ├── video_processor.py # Traitement vidéo et GIF
    ├── visualizer.py     # Dessin et annotations visuelles
    └── error_gallery.py  # Gestion des exemples d'erreurs
```

---

## Composants Clés

### 1. `app.py` (L'Interface)
C'est le chef d'orchestre. Il utilise **Gradio** pour créer l'interface utilisateur.
- **Fonctions `process_upload` / `process_video_upload`** : Font le lien entre l'UI et la logique métier.
- **Structure en onglets** : Image, Vidéo, Historique, Paramètres.
- **CSS Personnalisé** : Assure le look premium "sombre et moderne".

### 2. `utils/pipeline.py` (Le Cerveau)
Gère l'initialisation et l'exécution des modèles d'IA.
- **`ALPRPipeline`** : Classe principale qui charge YOLOv8 et `fast-plate-ocr`.
- **`reload_model(name)`** : Permet de changer de modèle sans redémarrer l'app (optimisation RAM).
- **`process_image()`** : Découpe l'image, détecte la plaque, l'isole et lance l'OCR.

### 3. `utils/access_control.py` (La Logique Métier)
Simule le système de barrière de parking.
- **`AccessController`** : Gère la "allowlist" (liste blanche).
- **`check_access(plate)`** : Vérifie si une plaque est autorisée (insensible à la casse).
- **`log_attempt()`** : Enregistre chaque passage dans `outputs/access_log.csv`.

### 4. `utils/video_processor.py` (Traitement Temporel)
Gère les fichiers lourds (MP4/GIF).
- **Mode Sample** : Extrait X images clés pour un résultat ultra-rapide.
- **Mode Annotate** : Recrée une nouvelle vidéo MP4 trame par trame avec les plaques surlignées.
- **Optimisation** : Utilise `cv2` (OpenCV) pour une manipulation efficace des flux vidéo.

### 5. `utils/visualizer.py` (Le Rendu)
Transforme les données brutes en visuels compréhensibles.
- Dessine les boîtes de détection colorées.
- Génère les bannières **"ACCÈS AUTORISÉ"** (Vert) ou **"ACCÈS REFUSÉ"** (Rouge).

---

## Flux de données (Image)

1. **Upload** : L'utilisateur dépose une image dans `app.py`.
2. **Détection** : `pipeline.py` demande à YOLOv8 : "Où est la plaque ?".
3. **Capture** : On découpe uniquement le rectangle de la plaque.
4. **Lecture** : `fast-plate-ocr` transforme l'image de la plaque en texte.
5. **Vérification** : `access_control.py` vérifie si ce texte est dans la liste blanche.
6. **Rendu** : `visualizer.py` dessine le résultat final.
7. **Affichage** : Gradio affiche l'image annotée et le statut de l'accès.

---

## Déploiement & Environnement

- **Python 3.10+**
- **Docker** : Utilise une image légère Debian pour garantir que l'environnement est identique partout.
- **RAM** : Optimisé pour tourner sous **16 GB** (Hugging Face Free Tier).
- **Persistance** : L'historique des accès est sauvegardé localement dans `outputs/`.

---

## Maintenance & Évolutions

- **Ajouter un modèle** : Placer le fichier `.pt` dans `demo/models/`. Il sera détecté automatiquement.
- **Modifier les accès** : Onglet "Settings" de l'application en direct (pas besoin de toucher au code).
- **Logs** : Télécharger `outputs/access_log.csv` pour analyse externe (Excel/Pandas).
