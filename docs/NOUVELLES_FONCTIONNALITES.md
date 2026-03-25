# Nouvelles Fonctionnalités (v2)

Ce document décrit les ajouts majeurs apportés à la version 2 de l'application ALPR, concentrés sur la vérification de conformité, l'optimisation des performances et l'amélioration continue (Active Learning).

---

## 1. Vérification de Conformité (Incohérence Système/Terrain)

Le système ne se contente plus de lire la plaque ; il vérifie désormais si le véhicule détecté correspond à celui enregistré dans la base de données pour ce résident.

### Fonctionnement
Lorsqu'une plaque est identifiée comme "Autorisée" :
1.  Le système interroge la base de données pour obtenir la marque du véhicule associée au résident (ex: "Audi").
2.  En parallèle, un modèle de classification analyse l'image pour déterminer la marque du véhicule présent (ex: "BMW").
3.  Les deux informations sont comparées.

### Gestion des Cas
-   **Correspondance Validée** : La marque détectée correspond à la marque attendue. L'accès est validé normalement.
-   **Incohérence Détectée** : La marque détectée est différente de la marque attendue (ex: Audi vs BMW).
    -   Un avertissement "MARQUE INCORRECTE" s'affiche.
    -   L'événement est consigné dans le journal des "Incohérences".
    -   L'accès reste techniquement "Autorisé" (car la plaque est bonne), mais l'alerte visuelle est orange.
-   **Marque Non Détectée** : Si le modèle ne parvient pas à identifier la marque (qualité d'image, angle), l'alerte "MARQUE NON DETECTEE" est levée pour signaler qu'une vérification visuelle est nécessaire.

### Journalisation
Un onglet spécifique "Incohérence Système/Terrain" dans l'interface d'administration permet de consulter l'historique de ces alertes, incluant :
-   La plaque concernée
-   Le nom du résident
-   La marque attendue vs la marque détectée
-   L'heure de l'incident

---

## 2. Apprentissage Actif (Active Learning)

Cette fonctionnalité permet d'améliorer le modèle d'intelligence artificielle au fil du temps en transformant les erreurs du système en données d'entraînement précieuses.

### Processus de Correction et Export
Dans l'onglet "Incohérence Système/Terrain", le gardien peut désormais agir sur les erreurs détectées :
1.  **Identifier l'erreur** : Repérer une ligne où la marque détectée est fausse.
2.  **Corriger** : Sélectionner la *vraie* marque du véhicule dans le menu déroulant.
3.  **Exporter** : Cliquer sur le bouton "Corriger & Envoyer au Dataset".

### Intégration Hugging Face
Le système est connecté à un dépôt de données (Dataset) sur la plateforme Hugging Face.
-   Une fois la correction validée, l'image du véhicule et le label corrigé sont automatiquement envoyés vers le cloud.
-   Cela permet aux Data Scientists de récupérer ces "cas difficiles" pour réentraîner le modèle et le rendre plus performant.

### Configuration
Un nouvel onglet **"Settings"** permet de configurer la connexion :
-   **Token** : Clé d'accès sécurisée (Write Token).
-   **Nom du Dataset** : Identifiant du dépôt (format `utilisateur/dataset`, ex: `philippetos/plaques`).

---

## 3. Recherche "Floue" (Fuzzy Matching)

Pour pallier les erreurs mineures de l'OCR (Reconnaissance Optique de Caractères), un algorithme de correspondance approximative a été implémenté.

### Problème
Auparavant, une erreur d'un seul caractère (ex: `80W...` au lieu de `80N...`) empêchait l'identification du résident, classant le véhicule comme "Inconnu" et rendant impossible la vérification de marque.

### Solution
Le système calcule désormais un score de similarité entre la plaque lue et les plaques de la base de données.
-   Si le score dépasse 85% de similarité, le système associe la lecture au résident le plus proche.
-   Cela permet de déclencher les alertes de marque même sur des images de qualité moyenne où l'OCR n'est pas parfait.

---

## 4. Optimisation des Performances Vidéo

Le traitement vidéo (MP4 et GIF) a été optimisé pour gérer la charge supplémentaire induite par la reconnaissance de marque.

### Option de Désactivation
Une option "Désactiver la reconnaissance de marque" est disponible dans l'onglet Vidéo (cochée par défaut).
-   **Activé (Case cochée)** : Le système ignore l'étape de classification de marque. Le traitement est 2 à 3 fois plus rapide. Idéal pour une simple extraction de plaques.
-   **Désactivé (Case décochée)** : Le système analyse la marque sur chaque image. Plus lent, mais nécessaire pour alimenter les rapports d'incohérence.

### Stabilité
La gestion de la base de données a été revue pour supporter l'afflux massif de requêtes générées par l'analyse vidéo frame-par-frame, évitant les erreurs de type "Database Locked".

---

## 5. Administration Avancée

L'interface d'administration a été enrichie pour offrir un contrôle total sans toucher au code ou à la base SQL.

-   **CRUD Résidents** : Ajout, Modification et Suppression de résidents directement via l'interface.
-   **Boutons d'Action Rapide** : Toggle d'accès (Oui/Non) et suppression par lot via des cases à cocher.
-   **Statistiques** : Tableau de bord affichant le nombre de résidents total, actifs, bloqués et abonnés.
