# Architecture du Modèle Maison (SimpleYOLO)

Ce document détaille les modifications et ajouts nécessaires pour intégrer le modèle d'immatriculation personnalisé (`modelemaison.pt`) dans l'application ALPR. Contrairement aux modèles YOLOv8 standards (Ultralytics), ce modèle utilise une architecture PyTorch spécifique définie "from scratch".

---

## 1. Création de l'Architecture (`utils/custom_yolo.py`)

Nous avons dû recréer l'architecture exacte utilisée lors de l'entraînement pour que PyTorch puisse charger les poids du fichier `.pt`.

### Composants clés :
- **`ConvBlock`** : Un bloc standard `Conv2d` + `BatchNorm2d` + `LeakyReLU`.
- **`SimpleYOLO`** : 
    - **Backbone** : 5 blocs convolutifs avec `MaxPool2d` pour réduire la résolution de 416x416 à 13x13.
    - **Head** : Une tête de détection finale produisant une sortie de 6 canaux (`objectness`, `x`, `y`, `w`, `h`, `class`).
- **`load_custom_model`** : Utilitaire pour instancier la classe et charger le `state_dict`.

---

## 2. Chargement Dynamique (`utils/pipeline.py`)

Le pipeline a été modifié pour détecter le nom du fichier et choisir la bonne méthode de chargement.

### Modifications dans `ALPRPipeline` :
- **Détection automatique** : Si le fichier se nomme `modelemaison.pt`, le pipeline utilise `load_custom_model()` au lieu de `YOLO()` (Ultralytics).
- **Drapeau `is_custom_model`** : Un booléen interne permet de savoir quelle logique d'inférence appliquer lors du traitement de l'image.

```python
if os.path.basename(self.model_path) == "modelemaison.pt":
    self.yolo_model = load_custom_model(self.model_path)
    self.is_custom_model = True
```

---

## 3. Inférence et Post-traitement Personnalisés

Le modèle maison renvoie un tenseur brut de dimension `(13, 13, 6)`, représentant une grille. Nous avons dû implémenter une logique spécifique pour convertir ces données en boîtes englobantes (bounding boxes) exploitables.

### Étapes de `_run_custom_inference` :
1. **Prétraitement** : Redimensionnement de l'image en 416x416 et normalisation [0, 1].
2. **Forward Pass** : Exécution du modèle pour obtenir la grille de prédiction.
3. **Décodage de la Grille (S×S)** :
    - Pour chaque cellule (13x13), vérifier le score d'objet (`objectness`).
    - Si > seuil, calculer les coordonnées relatives `xc`, `yc` par rapport à la cellule.
    - Convertir les dimensions `w`, `h` relatives.
4. **Rescaling** : Conversion des coordonnées normales (0-1) vers les pixels réels de l'image d'origine.
5. **Formatage** : Retourner une liste de dictionnaires compatibles avec le reste du pipeline OCR.

---

## 4. Résumé des fichiers impactés

| Fichier | Action | Rôle |
|:---|:---|:---|
| `utils/custom_yolo.py` | **Nouveau** | Définition de la classe `SimpleYOLO` et `ConvBlock`. |
| `utils/pipeline.py` | **Modifié** | Intégration du chargement et de la logique d'inférence dédiée. |
| `app.py` | **Inchangé** | L'interface Gradio utilise le pipeline sans savoir si le modèle est custom ou non (abstraction). |

---

> Cette approche permet à l'application de rester compatible avec les futurs modèles YOLOv8 standards tout en supportant votre recherche personnalisée.
