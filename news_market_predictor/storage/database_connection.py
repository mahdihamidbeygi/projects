"""
Database connection management with support for SQLite and PostgreSQL.
Implements connection pooling, error handling, and migration support.
"""

import os
import logging
import sqlite3
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Configuration for database connections."""

    def __init__(
        self,
        db_type: str = "sqlite",
        db_path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
    ):
        """
        Initialize database configuration.

        Args:
            db_type: Database type ('sqlite' or 'postgresql')
            db_path: Path to SQLite database file
            host: PostgreSQL host
            port: PostgreSQL port
            database: PostgreSQL database name
            user: PostgreSQL username
            password: PostgreSQL password
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_timeout: Pool timeout in seconds
        """
        self.db_type = db_type.lower()
        self.db_path = db_path or "news_market_predictor.db"
        self.host = host or "localhost"
        self.port = port or 5432
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables."""
        db_type = os.getenv("DB_TYPE", "sqlite")

        if db_type == "sqlite":
            return cls(
                db_type="sqlite",
                db_path=os.getenv("DB_PATH", "news_market_predictor.db"),
            )
        elif db_type == "postgresql":
            return cls(
                db_type="postgresql",
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
                pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def validate(self) -> bool:
        """Validate configuration."""
        if self.db_type not in ["sqlite", "postgresql"]:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        if self.db_type == "postgresql":
            if not all([self.database, self.user, self.password]):
                raise ValueError("PostgreSQL requires database, user, and password")

        return True


class DatabaseConnection:
    """
    Database connection manager with support for SQLite and PostgreSQL.
    Handles connection pooling, error recovery, and transaction management.
    """

    def __init__(self, config: DatabaseConfig):
        """Initialize database connection manager."""
        self.config = config
        self.config.validate()
        self._connection_pool = None
        self._engine = None
        self._setup_connection()

    def _setup_connection(self) -> None:
        """Setup database connection based on configuration."""
        if self.config.db_type == "sqlite":
            self._setup_sqlite()
        elif self.config.db_type == "postgresql":
            self._setup_postgresql()

    def _setup_sqlite(self) -> None:
        """Setup SQLite connection."""
        # Ensure directory exists
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using SQLite database at {self.config.db_path}")

    def _setup_postgresql(self) -> None:
        """Setup PostgreSQL connection with connection pooling."""
        try:
            import psycopg2
            from psycopg2 import pool

            self._connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=self.config.pool_size,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
            )

            logger.info(
                f"PostgreSQL connection pool created for {self.config.database}"
            )

        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. "
                "Install it with: pip install psycopg2-binary"
            )
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool.

        Yields:
            Database connection object

        Example:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        if self.config.db_type == "sqlite":
            conn = sqlite3.connect(self.config.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            try:
                yield conn
            finally:
                conn.close()

        elif self.config.db_type == "postgresql":
            if not self._connection_pool:
                raise RuntimeError("PostgreSQL connection pool not initialized")

            conn = self._connection_pool.getconn()
            try:
                yield conn
            finally:
                self._connection_pool.putconn(conn)

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        Automatically commits on success or rolls back on error.

        Example:
            with db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO table VALUES (?)", (value,))
        """
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction failed, rolling back: {e}")
                raise

    def execute_query(
        self, query: str, params: tuple = None, fetch_one: bool = False
    ) -> Union[list, tuple, None]:
        """
        Execute a query and return results.

        Args:
            query: SQL query string
            params: Query parameters
            fetch_one: If True, return only first result

        Returns:
            Query results or None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or ())

                if fetch_one:
                    return cursor.fetchone()
                else:
                    return cursor.fetchall()

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def execute_update(self, query: str, params: tuple = None) -> int:
        """
        Execute an update/insert/delete query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of affected rows
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or ())
                return cursor.rowcount

        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            raise

    def execute_many(self, query: str, params_list: list) -> int:
        """
        Execute a query with multiple parameter sets.

        Args:
            query: SQL query string
            params_list: List of parameter tuples

        Returns:
            Number of affected rows
        """
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                return cursor.rowcount

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self.config.db_type == "postgresql" and self._connection_pool:
            self._connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database.

        Returns:
            Dictionary with database information
        """
        info = {
            "type": self.config.db_type,
            "connected": self.test_connection(),
        }

        if self.config.db_type == "sqlite":
            info["path"] = self.config.db_path
            info["size_bytes"] = (
                Path(self.config.db_path).stat().st_size
                if Path(self.config.db_path).exists()
                else 0
            )

        elif self.config.db_type == "postgresql":
            info["host"] = self.config.host
            info["port"] = self.config.port
            info["database"] = self.config.database
            info["pool_size"] = self.config.pool_size

        return info

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
