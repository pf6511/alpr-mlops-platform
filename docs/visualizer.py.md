# Documentation Technique : `utils/visualizer.py`

`visualizer.py` est responsable de la transformation des données brutes (coordonnées, scores) en éléments visuels compréhensibles pour l'utilisateur.

## Fonctions de Rendu

### `annotate_detection(image, bbox, confidence)`
- **Rôle** : Dessine un rectangle (Bounding Box) autour de la plaque détectée.
- **Détail** : Utilise OpenCV (`cv2.rectangle`) et ajoute un badge de confiance au-dessus de la boîte.

### `create_confidence_badge(confidence)`
- **Rôle** : Détermine la couleur et l'emoji en fonction du score de confiance.
- **Seuils** :
    - **≥ 85%** : Vert (Confiance élevée)
    - **≥ 60%** : Orange (Confiance moyenne)
    - **< 60%** : Rouge (Confiance faible)

## Rapports & Étapes

### `create_analysis_report(results)`
Génère un rapport complet en **Markdown** qui résume :
- Les conditions d'image (Luminosité, Netteté).
- Le texte lu par l'OCR.
- Les scores de détection et de lecture.
- Des avertissements si les conditions sont trop dégradées pour un résultat fiable.

### `create_step_images(results)`
Prépare la liste d'images pour le carrousel visuel de Gradio.
1. **Entrée brute**
2. **Détection YOLOv8** (Image avec boîtes)
3. **Extraction ROI** (Zoom sur la plaque)
4. **Résultat final** (Image annotée avec le texte reconnu)

### `format_ocr_result(text, confidence)`
Mise en forme esthétique du résultat texte :
- Espacement des caractères (ex : `A B C 1 2 3`) pour une meilleure lisibilité.
- Badge de confiance dynamique.

---
*Ce fichier fait partie de la documentation technique du projet ALPR.*
