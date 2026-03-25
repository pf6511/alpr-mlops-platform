# Documentation Technique : `app.py`

`app.py` est le point d'entrée principal de l'application web. Il utilise la bibliothèque **Gradio** pour construire une interface interactive permettant aux utilisateurs de tester le système ALPR.

## Structure de l'Interface

L'interface est organisée en 4 onglets principaux :

1.  **Image Processing** : Analyse d'images fixes avec visualisation par étapes.
2.  **Video/GIF Processing** : Analyse de fichiers lourds avec modes "Sample" (rapide) ou "Annotate" (complet).
3.  **History** : Consultation des logs d'accès stockés dans `access_log.csv`.
4.  **Settings** : Configuration de la liste blanche (allowlist) et sélection dynamique du modèle IA.

## Fonctions Principales

### `process_upload(image, conf_threshold)`
- **Rôle** : Gère l'analyse d'une image unique.
- **Flux** : 
    1. Sauvegarde l'image temporairement.
    2. Appelle `pipeline.process_image`.
    3. Formate les résultats pour l'affichage Gradio (images des étapes + rapport texte).

### `process_video_upload(video, mode, conf, num_samples)`
- **Rôle** : Pilote le traitement des vidéos.
- **Modes** :
    - **Sample** : Appelle `sample_video_frames` pour un aperçu rapide.
    - **Annotate** : Appelle `create_annotated_video` pour une vidéo complète.
    - **GIF** : Appelle `process_gif`.
- **Logique d'accès** : Identifie les plaques uniques détectées et les enregistre dans l'historique via `access_controller`.

### `reload_model_handler(model_name)`
- **Rôle** : Permet de changer le modèle YOLOv8 en cours d'exécution.
- **Action** : Appelle `pipeline.reload_model` et renvoie un message de statut à l'utilisateur.

## Design & Expérience Utilisateur
- **CSS Personnalisé** : Définit un thème sombre, des boutons contrastés et une typographie moderne (Inter).
- **Feedback Visuel** : Utilise des bannières colorées (Vert/Rouge) pour indiquer l'autorisation d'accès directement dans l'interface.
- **Optimisation** : Désactive les exemples d'images lourds en version "Hugging Face" pour économiser la RAM.

---
*Ce fichier fait partie de la documentation technique du projet ALPR.*
