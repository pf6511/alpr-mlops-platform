# Documentation Technique : `utils/access_control.py`

`access_control.py` gère la logique métier spécifique au système de parking : qui a le droit d'entrer et laisser un historique.

## La Classe `AccessController`

### Gestion de la Liste Blanche (Allowlist)
- **Stockage** : Utilise un `set()` Python pour une recherche quasi-instantanée ($O(1)$).
- **`update(text_content)`** : Analyse un texte multi-lignes, nettoie chaque plaque et met à jour la liste en mémoire.
- **Défaut** : Contient une série de plaques de démonstration (ex: `51A77529`) pour tester immédiatement le système.

### Normalisation (`normalize`)
Pour éviter les erreurs dues à la casse ou aux symboles, chaque plaque subit un traitement rigoureux :
1. Conversion en **MAJUSCULES**.
2. Suppression de tous les caractères spéciaux (tirets, points, espaces).
3. Exemple : `"ab-123.cd"` et `"AB 123 CD"` deviennent tous deux `"AB123CD"`.

### Vérification (`check_access`)
Compare la plaque détectée par l'OCR avec la liste blanche.
- Renvoie un booléen et un message formaté avec des emojis (✅ ou ⛔).

### Journalisation (`log_attempt`)
Sauvegarde chaque tentative d'accès dans un fichier persistant : `outputs/access_log.csv`.
- **Colonnes** : `Timestamp`, `Plate`, `Status` (GRANTED/DENIED), `Normalized`.
- **Sécurité** : Crée le dossier `outputs/` et les en-têtes du fichier CSV s'ils n'existent pas.

---
*Ce fichier fait partie de la documentation technique du projet ALPR.*
