"""
Migration des résidents depuis le postgres local vers NeonDB.

Usage:
    python scripts/migrate_to_neon.py

Prérequis:
    pip install psycopg2-binary
    Le container alpr-postgres doit tourner (ou fournir DB_SOURCE_URL)
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

# ── Source : postgres local (container alpr-postgres) ────────────────────────
SOURCE_URL = os.environ.get(
    "DB_SOURCE_URL",
    "postgresql://alpr:alpr_secret@localhost:5432/alpr"
)

# ── Destination : NeonDB ─────────────────────────────────────────────────────
NEON_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_klAFr1m2uLgQ@ep-misty-frost-ag9y7m5y-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)


def connect(url: str, label: str):
    try:
        conn = psycopg2.connect(url)
        print(f"✅ Connecté à {label}")
        return conn
    except Exception as e:
        print(f"❌ Connexion {label} échouée: {e}")
        sys.exit(1)


def migrate_residents(src_conn, dst_conn):
    with src_conn.cursor(cursor_factory=RealDictCursor) as src:
        src.execute("SELECT * FROM residents ORDER BY id")
        rows = list(src.fetchall())

    if not rows:
        print("⚠️  Aucun résident trouvé dans la source")
        return 0

    print(f"📋 {len(rows)} résidents à migrer...")

    insert_sql = """
        INSERT INTO residents
            (plaque, nom, prenom, age, telephone, adresse, ville,
             code_postal, abonnement, acces, marque_declaree, created_at, updated_at)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (plaque) DO UPDATE SET
            nom            = EXCLUDED.nom,
            prenom         = EXCLUDED.prenom,
            age            = EXCLUDED.age,
            telephone      = EXCLUDED.telephone,
            adresse        = EXCLUDED.adresse,
            ville          = EXCLUDED.ville,
            code_postal    = EXCLUDED.code_postal,
            abonnement     = EXCLUDED.abonnement,
            acces          = EXCLUDED.acces,
            marque_declaree= EXCLUDED.marque_declaree,
            updated_at     = EXCLUDED.updated_at
    """

    ok = 0
    with dst_conn.cursor() as dst:
        for row in rows:
            try:
                dst.execute(insert_sql, (
                    row['plaque'], row['nom'], row['prenom'], row['age'],
                    row['telephone'], row['adresse'], row['ville'],
                    row['code_postal'], row['abonnement'], row['acces'],
                    row.get('marque_declaree'),
                    row.get('created_at'), row.get('updated_at'),
                ))
                ok += 1
            except Exception as e:
                print(f"  ⚠️  Erreur ligne {row['plaque']}: {e}")
        dst_conn.commit()

    return ok


def ensure_schema(conn):
    """Crée les tables si elles n'existent pas encore sur NeonDB."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS residents (
                id SERIAL PRIMARY KEY,
                plaque VARCHAR(20) UNIQUE NOT NULL,
                nom VARCHAR(100) NOT NULL,
                prenom VARCHAR(100) NOT NULL,
                age INTEGER NOT NULL,
                telephone VARCHAR(20) NOT NULL,
                adresse VARCHAR(255) NOT NULL,
                ville VARCHAR(100) NOT NULL,
                code_postal VARCHAR(10) NOT NULL,
                abonnement VARCHAR(3) NOT NULL CHECK(abonnement IN ('oui', 'non')),
                acces VARCHAR(3) NOT NULL CHECK(acces IN ('oui', 'non')),
                marque_declaree VARCHAR(50) DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id SERIAL PRIMARY KEY,
                plaque VARCHAR(20) NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                resultat VARCHAR(10) NOT NULL CHECK(resultat IN ('autorisé', 'refusé')),
                normalized_plate VARCHAR(20) NOT NULL,
                marque_predite VARCHAR(50) DEFAULT NULL,
                marque_confiance REAL DEFAULT NULL,
                mismatch BOOLEAN DEFAULT FALSE
            )
        """)
        conn.commit()
    print("✅ Schéma NeonDB vérifié")


if __name__ == "__main__":
    print("🚀 Migration vers NeonDB\n")

    src = connect(SOURCE_URL, "postgres local")
    dst = connect(NEON_URL,   "NeonDB")

    ensure_schema(dst)
    migrated = migrate_residents(src, dst)

    src.close()
    dst.close()

    print(f"\n✅ Migration terminée : {migrated} résidents migrés vers NeonDB")
