# Guide d'Administration et Base de Données

Ce document décrit le fonctionnement du système de base de données pour la gestion des résidents et le contrôle d'accès ALPR, ainsi que l'interface d'administration client.

---

## 1. Architecture de la Base de Données

Le système utilise **SQLite** pour stocker les données localement dans le fichier `demo/alpr.db`. Il gère deux tables principales :

### A. Table `residents` (Utilisateurs et Accès)
Stocke les informations des résidents et leurs droits d'accès.

| Champ | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER | Identifiant unique (Clé Primaire) |
| `plaque` | TEXT | Numéro de plaque d'immatriculation (Doit être unique) |
| `nom`, `prenom` | TEXT | Identité du résident |
| `acces` | TEXT | Droit d'entrée : `'oui'` ou `'non'` |
| `abonnement` | TEXT | Statut d'abonnement : `'oui'` ou `'non'` |
| Autres... | TEXT | Adresse, Ville, Téléphone, etc. |

### B. Table `logs` (Historique)
Stocke chaque tentative de détection de plaque.

| Champ | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | TEXT | Date et heure de la détection |
| `plaque` | TEXT | Plaque détectée (brute) |
| `normalized_plate`| TEXT | Plaque normalisée (sans tirets/espaces) |
| `resultat` | TEXT | `'autorisé'` ou `'refusé'` selon les droits au moment T |

---

## 2. Fonctionnement du Contrôle d'Accès

Le contrôle est géré par `demo/utils/access_control.py` en coordination avec `demo/utils/database.py`.

1.  **Synchronisation** : Au démarrage (et lors de modifications), le système charge en mémoire la liste blanche (`allowlist`) contenant uniquement les plaques des résidents ayant `acces = 'oui'`.
2.  **Normalisation** : Les plaques sont nettoyées (tout en majuscules, suppression des espaces et tirets) pour la comparaison.
    *   Ex: `AA-123-BB` -> `AA123BB`
3.  **Vérification** : Lorsqu'une plaque est détectée, elle est comparée à la liste blanche en mémoire.

---

## 3. Interface d'Administration Client

L'application Gradio dispose d'un onglet **"👥 Administration Clients"** permettant de gérer la base de données sans toucher au code ou aux fichiers SQL.

### Fonctionnalités Disponibles :

#### Recherche
*   Barre de recherche permettant de filtrer les résidents par **Plaque**, **Nom** ou **Prénom**.

#### Ajouter un Résident
*   Formulaire complet pour créer un nouveau dossier.
*   **Important** : Le champ `Plaque` doit être unique. La création échouera si la plaque existe déjà.

#### Liste et Actions de Masse
Le tableau principal permet d'interagir avec les dossiers existants :
*   **Édition Rapide** : Double-cliquez sur une cellule pour modifier une valeur (ex: changer un numéro de téléphone), puis cliquez sur **"💾 Sauvegarder Modifications"**.
*   **Suppression** : Cochez la case "Sélection" à gauche d'une ou plusieurs lignes, puis cliquez sur **"🗑️ Supprimer Cochés"**.
*   **Gestion d'Accès Rapide** : Cochez des lignes et cliquez sur **"🔄 Toggle Accès"** pour basculer rapidement leur droit d'entrée (Oui -> Non ou Non -> Oui).

#### Logs (Backend)
*   Visualisation des 50 dernières détections enregistrées par le système, avec le statut (Autorisé/Refusé).

---

## 4. Import CSV

Si la base de données est vide au démarrage, le système tente d'importer automatiquement les données depuis un fichier CSV situé à `../plaques_avec_donnees.csv`.
Ceci est géré par la méthode `import_from_csv` dans `database.py`.
