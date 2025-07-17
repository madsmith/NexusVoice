
from datetime import datetime
import logging
from pathlib import Path
import sqlite3
from nexusvoice.core.config import NexusConfig

class InteractionLog:
    def __init__(self, config: NexusConfig, log_root: Path):
        self.config: NexusConfig = config
        self.log_root: Path = log_root

        self._db_file: Path = self.log_root / "interaction_log.db"
        self._recording_path: Path = self.log_root / "recordings"
    
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_path()
    
    def _init_path(self):
        if not self.log_root.exists():
            self.log_root.mkdir(parents=True, exist_ok=True)
        if not self._db_file.exists():
            self._db_file.touch()
        if not self._recording_path.exists():
            self._recording_path.mkdir(parents=True, exist_ok=True)
    
    CURRENT_SCHEMA_VERSION = 1

    def init_db(self):
        conn = sqlite3.connect(self._db_file)

        cursor = conn.cursor()

        # Check if the info table exists to determine current schema version
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='info'")
        table_exists = cursor.fetchone() is not None
        
        current_version = 0
        if table_exists:
            # Get current schema version from the info table
            cursor.execute("SELECT value FROM info WHERE key='schema_version'")
            result = cursor.fetchone()
            if result:
                current_version = int(result[0])
        
        # Run all necessary migrations in sequence
        self._run_migrations(conn, current_version)
        
        conn.commit()
        conn.close()
    
    def _run_migrations(self, conn, current_version):
        """
        Run migrations sequentially to update the database schema.
        
        Args:
            conn: SQLite connection
            current_version: Current schema version of the database
        """
        # Dictionary of migration functions keyed by target version
        migrations = {
            1: self._migrate_to_v1,
        }
        
        # Apply migrations in version order
        for version in sorted([v for v in migrations.keys() if v > current_version]):
            self.logger.info(f"Migrating database schema to version {version}")
            migrations[version](conn)
            
            # Update schema version
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO info (key, value, updated_at) VALUES (?, ?, ?)",
                ("schema_version", str(version), datetime.now().isoformat())
            )
            conn.commit()
        
    def _migrate_to_v1(self, conn):
        cursor = conn.cursor()

        # Create info table for key-value pairs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS info (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table of wake word activations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wake_word_log (
                rowid INTEGER PRIMARY KEY ASC AUTOINCREMENT, 
                interaction_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                recording_file TEXT,
                transcription TEXT,
                wake_word TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interaction_log (
                rowid INTEGER PRIMARY KEY ASC AUTOINCREMENT, 
                interaction_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                recording_file TEXT,
                transcription TEXT
            )
        """)

    def record_wake_word(self, recording_file: Path, wake_word: str, transcription: str):
        filename = recording_file.name

        conn = sqlite3.connect(self._db_file)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO wake_word_log (interaction_id, recording_file, wake_word, transcription)
            VALUES (?, ?, ?, ?)
        """, ("", filename, wake_word, transcription))
        conn.commit()
        conn.close()

    def record_interaction(self, recording_file: Path, transcription: str):
        filename = recording_file.name
        
        conn = sqlite3.connect(self._db_file)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO interaction_log (interaction_id, recording_file, transcription)
            VALUES (?, ?, ?)
        """, ("", filename, transcription))
        conn.commit()
        conn.close()