# Guide de Vérification de Compatibilité des Modèles

Ce document explique comment vérifier et intégrer de nouveaux modèles YOLO personnalisés dans le pipeline ALPR.

## A. Types de Modèles Supportés

Le pipeline (`demo/utils/pipeline.py`) supporte deux formats de modèles :

1.  **Modèles Standard YOLOv8**
    *   **Format**: `.pt` générique exporté par Ultralytics.
    *   **Chargement**: `ultralytics.YOLO(path)`
    *   **Architecture**: Architecture standard YOLOv8 (nano, small, medium, etc.).

2.  **Modèles Personnalisés "SimpleYOLO"**
    *   **Fichiers supportés**: `modelemaison.pt`, `YOLO_From_Scratch_LicensePlatev2.pt` (noms hardcodés).
    *   **Chargement**: `demo/utils/custom_yolo.py -> load_custom_model(path)`
    *   **Architecture**: Une architecture CNN personnalisée définie dans `custom_yolo.py` (classe `SimpleYOLO`).

> Si votre modèle a été entraîné "from scratch" via le notebook `YOLO_From_Scratch_LicensePlate v2.ipynb`, il s'agit d'un modèle **SimpleYOLO**. Il **NE PEUT PAS** être chargé avec `ultralytics.YOLO`.

---

## B. Vérification de l'Architecture (Pour les modèles "From Scratch")

Si vous introduisez un nouveau modèle entraîné avec le notebook, vous devez vous assurer que la définition de classe `SimpleYOLO` dans `demo/utils/custom_yolo.py` correspond **exactement** à celle utilisée lors de l'entraînement.

### Points à vérifier :

1.  **Séquence des couches (Layers)** :
    *   Dans le notebook, vérifiez la définition de `self.head`. Contient-elle des `Dropout` ?
    *   Exemple de différence critique :
        *   *Notebook* : `Conv -> Dropout -> Conv -> Dropout -> Conv`
        *   *Demo App* : `Conv -> Conv -> Conv`
    *   **Conséquence** : Erreur de chargement (`KeyError` ou décalage de poids).

2.  **Dimensions d'entrée/sortie** :
    *   Le modèle attend-il du 416x416 ou autre chose ?
    *   La couche finale doit sortir `(batch, 13, 13, 6)` (pour SimpleYOLO standard).

### Script de Vérification

Utilisez ce script pour tester si un fichier `.pt` peut être chargé par le code actuel de la démo.

Créez un fichier `check_model.py` à la racine de `demo/` :

```python
import torch
import sys
import os

# Ajouter le chemin pour trouver les modules
sys.path.append(os.getcwd())

from utils.custom_yolo import SimpleYOLO, load_custom_model

MODEL_PATH = "models/MON_NOUVEAU_MODELE.pt"  # Remplacez par votre chemin

def verify():
    print(f"🔍 Tentative de chargement de : {MODEL_PATH}")
    
    try:
        # Essai 1: Charger comme SimpleYOLO
        print("1️⃣ Test architecture SimpleYOLO...")
        model = load_custom_model(MODEL_PATH)
        print("✅ SUCCÈS : Le modèle est compatible SimpleYOLO.")
        return
    except Exception as e:
        print(f"❌ Échec SimpleYOLO : {e}")

    try:
        # Essai 2: Charger comme YOLOv8 Standard
        print("\n2️⃣ Test architecture Ultralytics YOLOv8...")
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        print("✅ SUCCÈS : Le modèle est compatible YOLOv8 Standard.")
        return
    except Exception as e:
        print(f"❌ Échec YOLOv8 : {e}")

    print("\n⚠️ CONCLUSION : Le modèle n'est compatible avec aucune architecture connue.")
    print("Vérifiez custom_yolo.py pour voir s'il correspond au code d'entraînement.")

if __name__ == "__main__":
    verify()
```

---

## C. Intégration d'un Nouveau Nom de Fichier

Si vous avez un nouveau modèle "From Scratch" avec un **nouveau nom** (ex: `mon_modele_v3.pt`), vous devez modifier `demo/utils/pipeline.py` pour qu'il utilise le chargeur personnalisé.

### Fichier : `demo/utils/pipeline.py`

Cherchez la méthode `_load_models` (vers la ligne 55) et `reload_model` (vers la ligne 90).

Ajoutez votre nom de fichier à la liste des conditions :

```python
# AVANT
if model_name == "modelemaison.pt" or model_name == "YOLO_From_Scratch_LicensePlatev2.pt":
    # ... utilser load_custom_model ...

# APRÈS
if model_name in ["modelemaison.pt", "YOLO_From_Scratch_LicensePlatev2.pt", "mon_modele_v3.pt"]:
     # ... utilser load_custom_model ...
```

Sans cette modification, le pipeline tentera de charger le fichier avec `ultralytics.YOLO` par défaut, ce qui échouera pour les modèles "From Scratch".
