# Documentation Technique : `utils/pipeline.py`

`pipeline.py` contient la classe `ALPRPipeline`, qui constitue le moteur intelligent du projet. Elle orchestre la détection et la lecture des plaques.

## La Classe `ALPRPipeline`

### Initialisation (`__init__`)
Lors de la création, la classe :
1. Recherche automatiquement un modèle `best.pt` dans les dossiers standards.
2. Charge le modèle **YOLOv8** (détection).
3. Charge **fast-plate-ocr** (reconnaissance globale de caractères).

### Méthode `process_image(image_path, conf_threshold)`
C'est la fonction la plus complexe. Elle réalise le pipeline complet :
1. **Étape 1 (Pré-analyse)** : Estime les conditions d'éclairage et de flou.
2. **Étape 2 (Détection)** : Lance YOLOv8 pour trouver les coordonnées (bounding boxes) des plaques.
3. **Étape 3 (Extraction)** : Découpe l'image (ROI - Region of Interest) pour chaque plaque trouvée.
4. **Étape 4 (OCR)** : Passe chaque découpe au modèle OCR pour extraire le texte.
5. **Étape 5 (Fusion)** : Agrège les résultats dans un dictionnaire structuré contenant les images de chaque étape.

### Gestion des Modèles
- **`get_available_models()`** : Méthode statique qui scanne le dossier `models/` pour lister les fichiers `.pt` disponibles.
- **`reload_model(model_name)`** : Permet de switcher de modèle dynamiquement. Elle charge le nouveau modèle *avant* de décharger l'ancien pour garantir la stabilité de l'application.

## Analyse des Conditions
La méthode privée `_estimate_conditions` analyse l'histogramme de l'image pour déterminer si les conditions sont :
- **Claires** (Jour)
- **Sombres** (Nuit/Faible luminosité)
- **Floues** (Vitesse ou mise au point)

Cela permet de générer des avertissements automatiques si la confiance du résultat est faible.

---
*Ce fichier fait partie de la documentation technique du projet ALPR.*
