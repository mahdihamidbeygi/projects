"""
Database schema management and migration system.
Handles schema versioning, migrations, and upgrades.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .database_connection import DatabaseConnection

logger = logging.getLogger(__name__)


class Migration:
    """Represents a database migration."""

    def __init__(
        self,
        version: int,
        name: str,
        up_sql: str,
        down_sql: Optional[str] = None,
        description: str = "",
    ):
        """
        Initialize migration.

        Args:
            version: Migration version number
            name: Migration name
            up_sql: SQL to apply migration
            down_sql: SQL to rollback migration
            description: Migration description
        """
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.description = description


class SchemaManager:
    """Manages database schema and migrations."""

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize schema manager."""
        self.db = db_connection
        self.migrations = self._get_migrations()
        self._ensure_migration_table()

    def _ensure_migration_table(self) -> None:
        """Create migration tracking table if it doesn't exist."""
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                description TEXT
            )
        """

        try:
            self.db.execute_update(create_table_sql)
            logger.info("Migration tracking table ensured")
        except Exception as e:
            logger.error(f"Failed to create migration table: {e}")
            raise

    def _get_migrations(self) -> List[Migration]:
        """
        Define all database migrations.

        Returns:
            List of Migration objects
        """
        migrations = []

        # Migration 1: Initial schema
        migrations.append(
            Migration(
                version=1,
                name="initial_schema",
                description="Create initial database schema",
                up_sql="""
                    -- Articles table
                    CREATE TABLE IF NOT EXISTS articles (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        url TEXT NOT NULL,
                        published_at TEXT NOT NULL,
                        source TEXT NOT NULL,
                        category TEXT NOT NULL,
                        raw_metadata TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );

                    -- Sentiment analysis table
                    CREATE TABLE IF NOT EXISTS sentiment_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id TEXT NOT NULL,
                        sentiment_score REAL NOT NULL,
                        confidence REAL NOT NULL,
                        key_phrases TEXT NOT NULL,
                        market_tone TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (article_id) REFERENCES articles(id),
                        UNIQUE(article_id)
                    );

                    -- Extracted entities table
                    CREATE TABLE IF NOT EXISTS extracted_entities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_value TEXT NOT NULL,
                        relevance_score REAL NOT NULL,
                        context TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (article_id) REFERENCES articles(id)
                    );

                    -- Predictions table
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id TEXT NOT NULL,
                        stock_symbol TEXT NOT NULL,
                        impact_direction TEXT NOT NULL,
                        impact_magnitude REAL NOT NULL,
                        confidence_level REAL NOT NULL,
                        reasoning TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (article_id) REFERENCES articles(id),
                        UNIQUE(article_id, stock_symbol)
                    );

                    -- Outcomes table
                    CREATE TABLE IF NOT EXISTS outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prediction_id TEXT NOT NULL,
                        stock_symbol TEXT NOT NULL,
                        actual_direction TEXT NOT NULL,
                        actual_magnitude REAL NOT NULL,
                        price_change_percent REAL NOT NULL,
                        evaluation_date TEXT NOT NULL,
                        time_horizon_hours INTEGER NOT NULL,
                        UNIQUE(prediction_id, time_horizon_hours)
                    );

                    -- Accuracy metrics table
                    CREATE TABLE IF NOT EXISTS accuracy_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stock_symbol TEXT NOT NULL,
                        time_period_days INTEGER NOT NULL,
                        total_predictions INTEGER NOT NULL,
                        correct_predictions INTEGER NOT NULL,
                        accuracy_rate REAL NOT NULL,
                        average_confidence REAL NOT NULL,
                        calculated_at TEXT NOT NULL,
                        UNIQUE(stock_symbol, time_period_days, calculated_at)
                    );
                """,
                down_sql="""
                    DROP TABLE IF EXISTS accuracy_metrics;
                    DROP TABLE IF EXISTS outcomes;
                    DROP TABLE IF EXISTS predictions;
                    DROP TABLE IF EXISTS extracted_entities;
                    DROP TABLE IF EXISTS sentiment_analysis;
                    DROP TABLE IF EXISTS articles;
                """,
            )
        )

        # Migration 2: Add indexes
        migrations.append(
            Migration(
                version=2,
                name="add_indexes",
                description="Add indexes for better query performance",
                up_sql="""
                    CREATE INDEX IF NOT EXISTS idx_articles_published 
                        ON articles(published_at);
                    CREATE INDEX IF NOT EXISTS idx_articles_source 
                        ON articles(source);
                    CREATE INDEX IF NOT EXISTS idx_articles_category 
                        ON articles(category);
                    
                    CREATE INDEX IF NOT EXISTS idx_sentiment_article 
                        ON sentiment_analysis(article_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_entities_article 
                        ON extracted_entities(article_id);
                    CREATE INDEX IF NOT EXISTS idx_entities_type 
                        ON extracted_entities(entity_type);
                    CREATE INDEX IF NOT EXISTS idx_entities_value 
                        ON extracted_entities(entity_value);
                    
                    CREATE INDEX IF NOT EXISTS idx_predictions_stock_date 
                        ON predictions(stock_symbol, created_at);
                    CREATE INDEX IF NOT EXISTS idx_predictions_article 
                        ON predictions(article_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_outcomes_prediction 
                        ON outcomes(prediction_id);
                    CREATE INDEX IF NOT EXISTS idx_outcomes_stock 
                        ON outcomes(stock_symbol);
                    
                    CREATE INDEX IF NOT EXISTS idx_accuracy_stock 
                        ON accuracy_metrics(stock_symbol);
                """,
                down_sql="""
                    DROP INDEX IF EXISTS idx_articles_published;
                    DROP INDEX IF EXISTS idx_articles_source;
                    DROP INDEX IF EXISTS idx_articles_category;
                    DROP INDEX IF EXISTS idx_sentiment_article;
                    DROP INDEX IF EXISTS idx_entities_article;
                    DROP INDEX IF EXISTS idx_entities_type;
                    DROP INDEX IF EXISTS idx_entities_value;
                    DROP INDEX IF EXISTS idx_predictions_stock_date;
                    DROP INDEX IF EXISTS idx_predictions_article;
                    DROP INDEX IF EXISTS idx_outcomes_prediction;
                    DROP INDEX IF EXISTS idx_outcomes_stock;
                    DROP INDEX IF EXISTS idx_accuracy_stock;
                """,
            )
        )

        return migrations

    def get_current_version(self) -> int:
        """
        Get current schema version.

        Returns:
            Current version number, 0 if no migrations applied
        """
        try:
            result = self.db.execute_query(
                "SELECT MAX(version) FROM schema_migrations", fetch_one=True
            )

            if result and result[0] is not None:
                return result[0]
            return 0

        except Exception as e:
            logger.warning(f"Could not get current version: {e}")
            return 0

    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """
        Get list of applied migrations.

        Returns:
            List of migration records
        """
        try:
            rows = self.db.execute_query(
                "SELECT version, name, applied_at, description "
                "FROM schema_migrations ORDER BY version"
            )

            migrations = []
            for row in rows:
                migrations.append(
                    {
                        "version": row[0],
                        "name": row[1],
                        "applied_at": row[2],
                        "description": row[3],
                    }
                )

            return migrations

        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []

    def migrate_up(self, target_version: Optional[int] = None) -> bool:
        """
        Apply migrations up to target version.

        Args:
            target_version: Target version (None for latest)

        Returns:
            True if successful, False otherwise
        """
        current_version = self.get_current_version()
        target = target_version or max(m.version for m in self.migrations)

        if current_version >= target:
            logger.info(f"Already at version {current_version}, no migrations needed")
            return True

        logger.info(f"Migrating from version {current_version} to {target}")

        # Get migrations to apply
        to_apply = [m for m in self.migrations if current_version < m.version <= target]
        to_apply.sort(key=lambda m: m.version)

        for migration in to_apply:
            try:
                logger.info(f"Applying migration {migration.version}: {migration.name}")

                # Execute migration SQL
                with self.db.transaction() as conn:
                    cursor = conn.cursor()

                    # Split and execute each statement
                    for statement in migration.up_sql.split(";"):
                        statement = statement.strip()
                        if statement:
                            cursor.execute(statement)

                    # Record migration
                    cursor.execute(
                        """
                        INSERT INTO schema_migrations 
                        (version, name, applied_at, description)
                        VALUES (?, ?, ?, ?)
                    """,
                        (
                            migration.version,
                            migration.name,
                            datetime.now().isoformat(),
                            migration.description,
                        ),
                    )

                logger.info(f"Migration {migration.version} applied successfully")

            except Exception as e:
                logger.error(f"Migration {migration.version} failed: {e}")
                return False

        logger.info(f"Successfully migrated to version {target}")
        return True

    def migrate_down(self, target_version: int = 0) -> bool:
        """
        Rollback migrations to target version.

        Args:
            target_version: Target version to rollback to

        Returns:
            True if successful, False otherwise
        """
        current_version = self.get_current_version()

        if current_version <= target_version:
            logger.info(f"Already at or below version {target_version}")
            return True

        logger.info(f"Rolling back from version {current_version} to {target_version}")

        # Get migrations to rollback
        to_rollback = [
            m for m in self.migrations if target_version < m.version <= current_version
        ]
        to_rollback.sort(key=lambda m: m.version, reverse=True)

        for migration in to_rollback:
            if not migration.down_sql:
                logger.error(
                    f"Migration {migration.version} has no rollback SQL, cannot proceed"
                )
                return False

            try:
                logger.info(
                    f"Rolling back migration {migration.version}: {migration.name}"
                )

                # Execute rollback SQL
                with self.db.transaction() as conn:
                    cursor = conn.cursor()

                    # Split and execute each statement
                    for statement in migration.down_sql.split(";"):
                        statement = statement.strip()
                        if statement:
                            cursor.execute(statement)

                    # Remove migration record
                    cursor.execute(
                        "DELETE FROM schema_migrations WHERE version = ?",
                        (migration.version,),
                    )

                logger.info(f"Migration {migration.version} rolled back successfully")

            except Exception as e:
                logger.error(f"Rollback of migration {migration.version} failed: {e}")
                return False

        logger.info(f"Successfully rolled back to version {target_version}")
        return True

    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get current migration status.

        Returns:
            Dictionary with migration status information
        """
        current_version = self.get_current_version()
        latest_version = max(m.version for m in self.migrations)
        applied = self.get_applied_migrations()

        return {
            "current_version": current_version,
            "latest_version": latest_version,
            "up_to_date": current_version >= latest_version,
            "pending_migrations": latest_version - current_version,
            "applied_migrations": applied,
        }

    def reset_database(self) -> bool:
        """
        Reset database by rolling back all migrations.

        WARNING: This will delete all data!

        Returns:
            True if successful, False otherwise
        """
        logger.warning("Resetting database - all data will be lost!")
        return self.migrate_down(target_version=0)
