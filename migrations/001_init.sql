-- ═══════════════════════════════════════════════════════════════════════════════
-- ALPR Engine - Migration 001: Initial Schema
-- Base de données: PostgreSQL (Neon DB)
-- ═══════════════════════════════════════════════════════════════════════════════

-- ───────────────────────────────────────────────────────────────────────────────
-- TABLE: residents (whitelist des véhicules autorisés)
-- ───────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS residents (
    id SERIAL PRIMARY KEY,
    
    -- Identification véhicule
    plaque VARCHAR(20) UNIQUE NOT NULL,
    marque_declaree VARCHAR(50) DEFAULT NULL,  -- Marque déclarée par le client
    
    -- Informations propriétaire
    nom VARCHAR(100) NOT NULL,
    prenom VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL,
    telephone VARCHAR(20) NOT NULL,
    adresse VARCHAR(255) NOT NULL,
    ville VARCHAR(100) NOT NULL,
    code_postal VARCHAR(10) NOT NULL,
    
    -- Statut
    abonnement VARCHAR(3) NOT NULL DEFAULT 'non' CHECK(abonnement IN ('oui', 'non')),
    acces VARCHAR(3) NOT NULL DEFAULT 'non' CHECK(acces IN ('oui', 'non')),
    
    -- Métadonnées
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour recherche rapide par plaque
CREATE INDEX IF NOT EXISTS idx_residents_plaque ON residents(plaque);
CREATE INDEX IF NOT EXISTS idx_residents_acces ON residents(acces);


-- ───────────────────────────────────────────────────────────────────────────────
-- TABLE: logs (historique des accès)
-- ───────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS logs (
    id SERIAL PRIMARY KEY,
    
    -- Plaque détectée
    plaque VARCHAR(20) NOT NULL,
    normalized_plate VARCHAR(20) NOT NULL,  -- Plaque normalisée (sans tirets/espaces)
    
    -- Résultat
    resultat VARCHAR(10) NOT NULL CHECK(resultat IN ('autorisé', 'refusé')),
    
    -- Classification marque (branche 2)
    marque_predite VARCHAR(50) DEFAULT NULL,
    marque_confiance REAL DEFAULT NULL,
    mismatch BOOLEAN DEFAULT FALSE,  -- TRUE si marque_predite ≠ marque_declaree
    
    -- Timestamp
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index pour requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_logs_plaque ON logs(plaque);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_logs_mismatch ON logs(mismatch) WHERE mismatch = TRUE;


-- ───────────────────────────────────────────────────────────────────────────────
-- TABLE: dataset_labels (images labellisées pour retraining)
-- ───────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dataset_labels (
    id SERIAL PRIMARY KEY,
    
    -- Référence image
    image_path VARCHAR(500) NOT NULL,  -- Chemin S3 de l'image
    log_id INTEGER REFERENCES logs(id) ON DELETE SET NULL,
    
    -- Labels
    marque_predite VARCHAR(50),      -- Ce que le modèle a prédit
    marque_corrigee VARCHAR(50),     -- Ce que l'opérateur a corrigé
    confiance_prediction REAL,
    
    -- Statut validation
    status VARCHAR(20) DEFAULT 'pending' CHECK(status IN ('pending', 'validated', 'rejected', 'exported')),
    validated_by VARCHAR(100) DEFAULT NULL,
    validated_at TIMESTAMP DEFAULT NULL,
    
    -- Métadonnées
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour la file de validation
CREATE INDEX IF NOT EXISTS idx_dataset_status ON dataset_labels(status);
CREATE INDEX IF NOT EXISTS idx_dataset_created ON dataset_labels(created_at DESC);


-- ───────────────────────────────────────────────────────────────────────────────
-- TABLE: model_versions (historique des modèles déployés)
-- ───────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    
    -- Identification
    model_name VARCHAR(100) NOT NULL,  -- ex: efficientnet-brand-classifier
    version VARCHAR(50) NOT NULL,
    mlflow_run_id VARCHAR(100),
    
    -- Métriques
    accuracy REAL,
    dataset_size INTEGER,
    training_date TIMESTAMP,
    
    -- Statut déploiement
    status VARCHAR(20) DEFAULT 'trained' CHECK(status IN ('trained', 'staging', 'production', 'archived')),
    deployed_at TIMESTAMP DEFAULT NULL,
    
    -- Métadonnées
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_status ON model_versions(model_name, status);


-- ───────────────────────────────────────────────────────────────────────────────
-- FONCTION: update_updated_at()
-- Met à jour automatiquement le champ updated_at
-- ───────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger sur residents
DROP TRIGGER IF EXISTS trigger_residents_updated_at ON residents;
CREATE TRIGGER trigger_residents_updated_at
    BEFORE UPDATE ON residents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();


-- ───────────────────────────────────────────────────────────────────────────────
-- VUES UTILITAIRES
-- ───────────────────────────────────────────────────────────────────────────────

-- Vue: Statistiques globales
CREATE OR REPLACE VIEW v_statistics AS
SELECT
    (SELECT COUNT(*) FROM residents) AS total_residents,
    (SELECT COUNT(*) FROM residents WHERE acces = 'oui') AS active_residents,
    (SELECT COUNT(*) FROM residents WHERE acces = 'non') AS blocked_residents,
    (SELECT COUNT(*) FROM residents WHERE abonnement = 'oui') AS subscribers,
    (SELECT COUNT(*) FROM logs WHERE DATE(timestamp) = CURRENT_DATE) AS logs_today,
    (SELECT COUNT(*) FROM logs WHERE mismatch = TRUE) AS total_mismatches,
    (SELECT COUNT(*) FROM dataset_labels WHERE status = 'pending') AS pending_labels;

-- Vue: Logs récents avec info résident
CREATE OR REPLACE VIEW v_logs_enriched AS
SELECT 
    l.id,
    l.plaque,
    l.normalized_plate,
    l.resultat,
    l.marque_predite,
    l.marque_confiance,
    l.mismatch,
    l.timestamp,
    r.nom,
    r.prenom,
    r.marque_declaree
FROM logs l
LEFT JOIN residents r ON l.normalized_plate = REPLACE(REPLACE(r.plaque, '-', ''), ' ', '')
ORDER BY l.timestamp DESC;

-- Vue: Mismatches en attente de validation
CREATE OR REPLACE VIEW v_pending_validation AS
SELECT 
    l.id AS log_id,
    l.plaque,
    l.marque_predite,
    l.marque_confiance,
    l.timestamp,
    r.marque_declaree,
    d.id AS label_id,
    d.image_path,
    d.status AS label_status
FROM logs l
LEFT JOIN residents r ON l.normalized_plate = REPLACE(REPLACE(r.plaque, '-', ''), ' ', '')
LEFT JOIN dataset_labels d ON d.log_id = l.id
WHERE l.mismatch = TRUE
ORDER BY l.timestamp DESC;


-- ───────────────────────────────────────────────────────────────────────────────
-- DONNÉES INITIALES (optionnel)
-- ───────────────────────────────────────────────────────────────────────────────

-- Insérer un résident de test (décommenter si nécessaire)
-- INSERT INTO residents (plaque, nom, prenom, age, telephone, adresse, ville, code_postal, abonnement, acces, marque_declaree)
-- VALUES ('AA-123-BB', 'Dupont', 'Jean', 35, '0600000000', '1 Rue de Paris', 'Paris', '75001', 'oui', 'oui', 'Renault');


-- ───────────────────────────────────────────────────────────────────────────────
-- FIN MIGRATION 001
-- ───────────────────────────────────────────────────────────────────────────────

-- Pour exécuter cette migration sur Neon DB:
-- psql "postgresql://user:password@ep-xxx.eu-central-1.aws.neon.tech/alpr?sslmode=require" -f 001_init.sql
