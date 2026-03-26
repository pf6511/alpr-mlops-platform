"""
ALPR Engine - Database Manager
Support SQLite (dev) et PostgreSQL/Neon DB (production)
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from contextlib import contextmanager
from pathlib import Path

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import get_settings

# PostgreSQL support (optionnel)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ psycopg2 non installé - Mode SQLite uniquement")


class DatabaseManager:
    
    def __init__(self, db_path: Optional[str] = None):
        self.settings = get_settings()
        self.db_config = self.settings.database

        # Support DATABASE_URL direct (NeonDB, Heroku, etc.)
        self.db_url = os.environ.get('DATABASE_URL')

        if db_path:
            self.mode = "sqlite"
            self.sqlite_path = db_path
            self.db_url = None
        elif self.db_url and POSTGRES_AVAILABLE:
            self.mode = "postgres"
            self.sqlite_path = None
        elif self.db_config.mode == "postgres" and POSTGRES_AVAILABLE:
            self.mode = "postgres"
            self.sqlite_path = None
        else:
            self.mode = "sqlite"
            self.sqlite_path = self.db_config.sqlite_path

        if self.mode == "postgres":
            host_info = self.db_url.split("@")[1].split("/")[0] if self.db_url else self.db_config.host
            print(f"📦 Database: POSTGRES → {host_info}")
        else:
            print(f"📦 Database: SQLITE → {self.sqlite_path}")

        # ✅ FIX CRITIQUE : créer le dossier SQLite
        if self.mode == "sqlite" and self.sqlite_path != ":memory:":
            sqlite_file = Path(self.sqlite_path)
            sqlite_file.parent.mkdir(parents=True, exist_ok=True)

        self._init_tables()
        
        if self.mode == "sqlite":
            self._import_csv_if_empty()

    # ═══════════════════════════════════════════════════════════════════════════
    # CONNECTION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    @contextmanager
    def get_connection(self):
        conn = None
        try:
            if self.mode == "postgres":
                if self.db_url:
                    conn = psycopg2.connect(self.db_url)
                else:
                    conn = psycopg2.connect(
                        host=self.db_config.host,
                        port=self.db_config.port,
                        dbname=self.db_config.name,
                        user=self.db_config.user,
                        password=self.db_config.password,
                        sslmode=self.db_config.sslmode
                    )
            else:
                # ✅ FIX CRITIQUE : garantir dossier avant connexion
                if self.sqlite_path != ":memory:":
                    sqlite_file = Path(self.sqlite_path)
                    sqlite_file.parent.mkdir(parents=True, exist_ok=True)
                    conn = sqlite3.connect(str(sqlite_file))
                else:
                    conn = sqlite3.connect(":memory:")

                conn.row_factory = sqlite3.Row

            yield conn
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def get_cursor(self, commit: bool = False):
        with self.get_connection() as conn:
            if self.mode == "postgres":
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()
    
    def _placeholder(self) -> str:
        return "%s" if self.mode == "postgres" else "?"
    
    def _row_to_dict(self, row) -> Dict:
        if row is None:
            return {}
        return dict(row)

    # ═══════════════════════════════════════════════════════════════════════════
    # SCHEMA INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _init_tables(self):

        if self.mode == "postgres":
            residents_sql = """
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
            """
            logs_sql = """
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
            """
        else:
            residents_sql = """
                CREATE TABLE IF NOT EXISTS residents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plaque TEXT UNIQUE NOT NULL,
                    nom TEXT NOT NULL,
                    prenom TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    telephone TEXT NOT NULL,
                    adresse TEXT NOT NULL,
                    ville TEXT NOT NULL,
                    code_postal TEXT NOT NULL,
                    abonnement TEXT NOT NULL CHECK(abonnement IN ('oui', 'non')),
                    acces TEXT NOT NULL CHECK(acces IN ('oui', 'non')),
                    marque_declaree TEXT DEFAULT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            logs_sql = """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plaque TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    resultat TEXT NOT NULL CHECK(resultat IN ('autorisé', 'refusé')),
                    normalized_plate TEXT NOT NULL,
                    marque_predite TEXT DEFAULT NULL,
                    marque_confiance REAL DEFAULT NULL,
                    mismatch INTEGER DEFAULT 0
                )
            """

        with self.get_cursor(commit=True) as cursor:
            cursor.execute(residents_sql)
            cursor.execute(logs_sql)

        print("✅ Tables initialisées")

    # (le reste de ton code ne change pas)
    
    def _import_csv_if_empty(self):
        """Importe le CSV initial si la table residents est vide."""
        # Vérifier si table vide
        with self.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM residents")
            count = cursor.fetchone()[0] if self.mode == "sqlite" else cursor.fetchone()['count']
        
        if count > 0:
            return
        
        # Chercher le CSV
        csv_paths = [
            Path(__file__).parent.parent.parent / "base_residents.csv",
            Path(__file__).parent.parent / "base_residents.csv",
            Path(__file__).parent.parent / "data" / "base_residents.csv",
        ]
        
        csv_path = None
        for p in csv_paths:
            if p.exists():
                csv_path = p
                break
        
        if not csv_path:
            print("⚠️ Pas de CSV initial trouvé")
            return
        
        # Importer le CSV
        try:
            import csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                return
            
            ph = self._placeholder()
            sql = f"""
                INSERT INTO residents 
                (plaque, nom, prenom, age, telephone, adresse, ville, code_postal, abonnement, acces)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
            """
            
            with self.get_cursor(commit=True) as cursor:
                for row in rows:
                    try:
                        cursor.execute(sql, (
                            row.get('plaque', ''),
                            row.get('nom', ''),
                            row.get('prenom', ''),
                            int(row.get('age', 0)),
                            row.get('telephone', ''),
                            row.get('adresse', ''),
                            row.get('ville', ''),
                            row.get('code_postal', ''),
                            row.get('abonnement', 'non'),
                            row.get('acces', 'non')
                        ))
                    except Exception as e:
                        print(f"⚠️ Erreur import ligne: {e}")
            
            print(f"✅ {len(rows)} résidents importés depuis CSV")
        except Exception as e:
            print(f"⚠️ Erreur import CSV: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESIDENTS CRUD
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_all_residents(self) -> List[Dict]:
        """Récupère tous les résidents."""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM residents ORDER BY nom, prenom")
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def search_residents(self, query: str = "") -> List[Dict]:
        """Recherche des résidents par plaque, nom ou prénom."""
        if not query:
            return self.get_all_residents()
        
        ph = self._placeholder()
        pattern = f"%{query}%"
        
        sql = f"""
            SELECT * FROM residents 
            WHERE plaque LIKE {ph} OR nom LIKE {ph} OR prenom LIKE {ph}
            ORDER BY nom, prenom
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(sql, (pattern, pattern, pattern))
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def get_resident_by_plate(self, plaque: str) -> Optional[Dict]:
        """Récupère un résident par sa plaque."""
        # Normaliser la plaque (supprimer espaces et tirets)
        plaque_normalized = plaque.replace(' ', '').replace('-', '').upper()
        
        ph = self._placeholder()
        with self.get_cursor() as cursor:
            # Comparer avec la version normalisée de la plaque en base
            if self.mode == "postgres":
                cursor.execute(
                    f"SELECT * FROM residents WHERE REPLACE(REPLACE(UPPER(plaque), ' ', ''), '-', '') = {ph}",
                    (plaque_normalized,)
                )
            else:
                cursor.execute(
                    f"SELECT * FROM residents WHERE REPLACE(REPLACE(UPPER(plaque), ' ', ''), '-', '') = {ph}",
                    (plaque_normalized,)
                )
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None
    
    def add_resident(self, data: Dict) -> Tuple[bool, str]:
        """Ajoute un nouveau résident."""
        ph = self._placeholder()
        sql = f"""
            INSERT INTO residents 
            (plaque, nom, prenom, age, telephone, adresse, ville, code_postal, abonnement, acces, marque_declaree)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
        """
        
        try:
            with self.get_cursor(commit=True) as cursor:
                cursor.execute(sql, (
                    data.get('plaque', ''),
                    data.get('nom', ''),
                    data.get('prenom', ''),
                    int(data.get('age', 0)),
                    data.get('telephone', ''),
                    data.get('adresse', ''),
                    data.get('ville', ''),
                    data.get('code_postal', ''),
                    data.get('abonnement', 'non'),
                    data.get('acces', 'non'),
                    data.get('marque_declaree', None)
                ))
            return True, f"✅ Résident {data.get('plaque')} ajouté"
        except Exception as e:
            if "UNIQUE" in str(e).upper() or "unique" in str(e).lower():
                return False, f"❌ Plaque {data.get('plaque')} déjà existante"
            return False, f"❌ Erreur: {e}"
    
    def update_resident(self, resident_id: int, data: Dict) -> Tuple[bool, str]:
        """Met à jour un résident."""
        ph = self._placeholder()
        sql = f"""
            UPDATE residents SET
                plaque = {ph},
                nom = {ph},
                prenom = {ph},
                age = {ph},
                telephone = {ph},
                adresse = {ph},
                ville = {ph},
                code_postal = {ph},
                abonnement = {ph},
                acces = {ph},
                marque_declaree = {ph},
                updated_at = CURRENT_TIMESTAMP
            WHERE id = {ph}
        """
        
        try:
            with self.get_cursor(commit=True) as cursor:
                cursor.execute(sql, (
                    data.get('plaque', ''),
                    data.get('nom', ''),
                    data.get('prenom', ''),
                    int(data.get('age', 0)),
                    data.get('telephone', ''),
                    data.get('adresse', ''),
                    data.get('ville', ''),
                    data.get('code_postal', ''),
                    data.get('abonnement', 'non'),
                    data.get('acces', 'non'),
                    data.get('marque_declaree', None),
                    resident_id
                ))
            return True, f"✅ Résident #{resident_id} mis à jour"
        except Exception as e:
            return False, f"❌ Erreur: {e}"
    
    def delete_resident(self, resident_id: int) -> Tuple[bool, str]:
        """Supprime un résident."""
        ph = self._placeholder()
        try:
            with self.get_cursor(commit=True) as cursor:
                cursor.execute(f"DELETE FROM residents WHERE id = {ph}", (resident_id,))
            return True, f"✅ Résident #{resident_id} supprimé"
        except Exception as e:
            return False, f"❌ Erreur: {e}"
    
    def toggle_access(self, resident_id: int) -> Tuple[bool, str, str]:
        """Inverse l'accès d'un résident."""
        ph = self._placeholder()
        
        # Récupérer l'état actuel
        with self.get_cursor() as cursor:
            cursor.execute(f"SELECT acces FROM residents WHERE id = {ph}", (resident_id,))
            row = cursor.fetchone()
            if not row:
                return False, "❌ Résident non trouvé", ""
            
            current = row['acces'] if self.mode == "postgres" else row[0]
            new_status = 'non' if current == 'oui' else 'oui'
        
        # Mettre à jour
        try:
            with self.get_cursor(commit=True) as cursor:
                cursor.execute(
                    f"UPDATE residents SET acces = {ph}, updated_at = CURRENT_TIMESTAMP WHERE id = {ph}",
                    (new_status, resident_id)
                )
            return True, f"✅ Accès changé → {new_status}", new_status
        except Exception as e:
            return False, f"❌ Erreur: {e}", ""
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOGS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_log(self, plaque: str, authorized: bool, normalized: str,
                marque_predite: str = None, marque_confiance: float = None,
                mismatch: bool = False) -> bool:
        """Ajoute une entrée de log."""
        ph = self._placeholder()
        timestamp = datetime.now().isoformat()
        resultat = "autorisé" if authorized else "refusé"
        
        sql = f"""
            INSERT INTO logs (plaque, timestamp, resultat, normalized_plate, marque_predite, marque_confiance, mismatch)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
        """
        
        try:
            with self.get_cursor(commit=True) as cursor:
                mismatch_val = mismatch if self.mode == "postgres" else (1 if mismatch else 0)
                cursor.execute(sql, (
                    plaque, timestamp, resultat, normalized,
                    marque_predite, marque_confiance, mismatch_val
                ))
            return True
        except Exception as e:
            print(f"⚠️ Erreur log: {e}")
            return False
    
    def get_logs(self, limit: int = 100) -> List[Dict]:
        """Récupère les derniers logs."""
        ph = self._placeholder()
        
        with self.get_cursor() as cursor:
            cursor.execute(f"SELECT * FROM logs ORDER BY timestamp DESC LIMIT {limit}")
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def get_mismatch_logs(self, limit: int = 100) -> List[Dict]:
        """Récupère les logs avec mismatch marque (pour labellisation)."""
        ph = self._placeholder()
        mismatch_val = "TRUE" if self.mode == "postgres" else "1"
        
        with self.get_cursor() as cursor:
            cursor.execute(f"""
                SELECT * FROM logs 
                WHERE mismatch = {mismatch_val}
                ORDER BY timestamp DESC 
                LIMIT {limit}
            """)
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques globales."""
        stats = {
            'total': 0,
            'active': 0,
            'blocked': 0,
            'subscribers': 0,
            'logs_today': 0,
            'mismatches_pending': 0
        }
        
        with self.get_cursor() as cursor:
            # Résidents
            cursor.execute("SELECT COUNT(*) FROM residents")
            row = cursor.fetchone()
            stats['total'] = row['count'] if self.mode == "postgres" else row[0]
            
            cursor.execute("SELECT COUNT(*) FROM residents WHERE acces = 'oui'")
            row = cursor.fetchone()
            stats['active'] = row['count'] if self.mode == "postgres" else row[0]
            
            cursor.execute("SELECT COUNT(*) FROM residents WHERE acces = 'non'")
            row = cursor.fetchone()
            stats['blocked'] = row['count'] if self.mode == "postgres" else row[0]
            
            cursor.execute("SELECT COUNT(*) FROM residents WHERE abonnement = 'oui'")
            row = cursor.fetchone()
            stats['subscribers'] = row['count'] if self.mode == "postgres" else row[0]
            
            # Logs aujourd'hui
            today = datetime.now().strftime('%Y-%m-%d')
            ph = self._placeholder()
            cursor.execute(f"SELECT COUNT(*) FROM logs WHERE timestamp LIKE {ph}", (f"{today}%",))
            row = cursor.fetchone()
            stats['logs_today'] = row['count'] if self.mode == "postgres" else row[0]
            
            # Mismatches en attente
            mismatch_val = "TRUE" if self.mode == "postgres" else "1"
            cursor.execute(f"SELECT COUNT(*) FROM logs WHERE mismatch = {mismatch_val}")
            row = cursor.fetchone()
            stats['mismatches_pending'] = row['count'] if self.mode == "postgres" else row[0]
        
        return stats
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WHITELIST (pour AccessController)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_whitelist(self) -> List[str]:
        """Récupère la liste des plaques autorisées."""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT plaque FROM residents WHERE acces = 'oui'")
            rows = cursor.fetchall()
            if self.mode == "postgres":
                return [row['plaque'] for row in rows]
            else:
                return [row[0] for row in rows]
    
    def get_all_residents(self) -> List[Dict]:
        """Récupère tous les résidents."""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM residents ORDER BY id DESC LIMIT 50")
            rows = cursor.fetchall()
            
            if self.mode == "postgres":
                return [dict(row) for row in rows]
            else:
                # SQLite - convertir les tuples en dicts
                columns = ['id', 'plaque', 'nom', 'prenom', 'age', 'telephone', 
                        'adresse', 'ville', 'code_postal', 'abonnement', 'acces', 
                        'marque_declaree', 'created_at', 'updated_at']
                return [dict(zip(columns, row)) for row in rows]
        
    def get_plate_with_brand(self, plaque: str) -> Optional[Dict]:
        """Récupère une plaque avec sa marque déclarée."""
        resident = self.get_resident_by_plate(plaque)
        if resident:
            return {
                'plaque': resident['plaque'],
                'marque_declaree': resident.get('marque_declaree'),
                'acces': resident['acces'],
                'nom': resident['nom'],
                'prenom': resident['prenom']
            }
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Test DatabaseManager...")
    
    # Test avec SQLite temporaire
    import tempfile
    test_db = tempfile.mktemp(suffix=".db")
    
    db = DatabaseManager(db_path=test_db)
    
    # Test CRUD
    success, msg = db.add_resident({
        'plaque': 'TEST-123-AB',
        'nom': 'Dupont',
        'prenom': 'Jean',
        'age': 35,
        'telephone': '0600000000',
        'adresse': '123 Rue Test',
        'ville': 'Paris',
        'code_postal': '75001',
        'abonnement': 'oui',
        'acces': 'oui',
        'marque_declaree': 'Renault'
    })
    print(f"Add: {msg}")
    
    # Search
    results = db.search_residents("TEST")
    print(f"Search: {len(results)} résultat(s)")
    
    # Stats
    stats = db.get_statistics()
    print(f"Stats: {stats}")
    
    # Log avec mismatch
    db.add_log("TEST-123-AB", True, "TEST123AB", "Peugeot", 0.85, mismatch=True)
    logs = db.get_mismatch_logs()
    print(f"Mismatch logs: {len(logs)}")
    
    # Cleanup
    os.remove(test_db)
    print("✅ Tests OK")
