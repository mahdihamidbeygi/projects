"""
Command-line interface for database management.
Provides tools for migrations, backups, and database operations.
"""

import sys
import argparse
import logging
from pathlib import Path

from .database_connection import DatabaseConnection, DatabaseConfig
from .schema_manager import SchemaManager
from .backup_manager import BackupManager, BackupConfig
from .data_access_layer import DataAccessLayer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_database(args):
    """Initialize database with schema."""
    try:
        config = (
            DatabaseConfig.from_env()
            if args.use_env
            else DatabaseConfig(db_type=args.db_type, db_path=args.db_path)
        )

        db = DatabaseConnection(config)
        schema_manager = SchemaManager(db)

        logger.info("Initializing database...")
        success = schema_manager.migrate_up()

        if success:
            logger.info("Database initialized successfully")
            status = schema_manager.get_migration_status()
            logger.info(f"Current version: {status['current_version']}")
            return 0
        else:
            logger.error("Database initialization failed")
            return 1

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return 1


def migrate_database(args):
    """Run database migrations."""
    try:
        config = (
            DatabaseConfig.from_env()
            if args.use_env
            else DatabaseConfig(db_type=args.db_type, db_path=args.db_path)
        )

        db = DatabaseConnection(config)
        schema_manager = SchemaManager(db)

        if args.direction == "up":
            logger.info(f"Migrating up to version {args.version or 'latest'}...")
            success = schema_manager.migrate_up(args.version)
        else:
            logger.info(f"Migrating down to version {args.version}...")
            success = schema_manager.migrate_down(args.version)

        if success:
            logger.info("Migration completed successfully")
            status = schema_manager.get_migration_status()
            logger.info(f"Current version: {status['current_version']}")
            return 0
        else:
            logger.error("Migration failed")
            return 1

    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return 1


def migration_status(args):
    """Show migration status."""
    try:
        config = (
            DatabaseConfig.from_env()
            if args.use_env
            else DatabaseConfig(db_type=args.db_type, db_path=args.db_path)
        )

        db = DatabaseConnection(config)
        schema_manager = SchemaManager(db)

        status = schema_manager.get_migration_status()

        print("\n=== Migration Status ===")
        print(f"Current Version: {status['current_version']}")
        print(f"Latest Version: {status['latest_version']}")
        print(f"Up to Date: {status['up_to_date']}")
        print(f"Pending Migrations: {status['pending_migrations']}")

        if status["applied_migrations"]:
            print("\nApplied Migrations:")
            for migration in status["applied_migrations"]:
                print(
                    f"  - Version {migration['version']}: {migration['name']} "
                    f"(applied {migration['applied_at']})"
                )

        return 0

    except Exception as e:
        logger.error(f"Error getting migration status: {e}")
        return 1


def create_backup(args):
    """Create database backup."""
    try:
        config = (
            DatabaseConfig.from_env()
            if args.use_env
            else DatabaseConfig(db_type=args.db_type, db_path=args.db_path)
        )

        if config.db_type != "sqlite":
            logger.error("Backup currently only supported for SQLite databases")
            return 1

        backup_config = BackupConfig(
            backup_dir=args.backup_dir,
            compress_backups=args.compress,
            verify_backups=args.verify,
        )

        backup_manager = BackupManager(config.db_path, backup_config)

        logger.info("Creating backup...")
        result = backup_manager.create_backup(force=True)

        if result["status"] == "completed":
            logger.info(f"Backup created successfully: {result['backup_file']}")
            if "compressed_file" in result:
                logger.info(f"Compressed to: {result['compressed_file']}")
            return 0
        else:
            logger.error(f"Backup failed: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return 1


def list_backups(args):
    """List available backups."""
    try:
        config = (
            DatabaseConfig.from_env()
            if args.use_env
            else DatabaseConfig(db_type=args.db_type, db_path=args.db_path)
        )

        if config.db_type != "sqlite":
            logger.error("Backup currently only supported for SQLite databases")
            return 1

        backup_config = BackupConfig(backup_dir=args.backup_dir)
        backup_manager = BackupManager(config.db_path, backup_config)

        backups = backup_manager.list_backups()

        print("\n=== Available Backups ===")
        if not backups:
            print("No backups found")
        else:
            for backup in backups:
                size_mb = backup["size_bytes"] / (1024 * 1024)
                compressed = " (compressed)" if backup["compressed"] else ""
                print(
                    f"  - {backup['filename']}: {size_mb:.2f} MB, "
                    f"created {backup['created_at']}{compressed}"
                )

        return 0

    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        return 1


def restore_backup(args):
    """Restore database from backup."""
    try:
        config = (
            DatabaseConfig.from_env()
            if args.use_env
            else DatabaseConfig(db_type=args.db_type, db_path=args.db_path)
        )

        if config.db_type != "sqlite":
            logger.error("Backup currently only supported for SQLite databases")
            return 1

        backup_config = BackupConfig(backup_dir=args.backup_dir)
        backup_manager = BackupManager(config.db_path, backup_config)

        logger.info(f"Restoring from backup: {args.backup_file}")
        result = backup_manager.restore_backup(args.backup_file)

        if result["status"] == "completed":
            logger.info("Database restored successfully")
            return 0
        else:
            logger.error(f"Restore failed: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        return 1


def database_stats(args):
    """Show database statistics."""
    try:
        config = (
            DatabaseConfig.from_env()
            if args.use_env
            else DatabaseConfig(db_type=args.db_type, db_path=args.db_path)
        )

        db = DatabaseConnection(config)
        dal = DataAccessLayer(db)

        stats = dal.get_database_stats()

        print("\n=== Database Statistics ===")
        print(f"Total Articles: {stats.get('total_articles', 0)}")
        print(f"Total Predictions: {stats.get('total_predictions', 0)}")
        print(f"Total Outcomes: {stats.get('total_outcomes', 0)}")
        print(f"Total Accuracy Records: {stats.get('total_accuracy_records', 0)}")
        print(f"Unique Stocks: {stats.get('unique_stocks', 0)}")

        db_info = db.get_database_info()
        print(f"\nDatabase Type: {db_info['type']}")
        print(f"Connected: {db_info['connected']}")

        if db_info["type"] == "sqlite":
            size_mb = db_info.get("size_bytes", 0) / (1024 * 1024)
            print(f"Database Size: {size_mb:.2f} MB")

        return 0

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Database management CLI for News Market Predictor"
    )

    parser.add_argument(
        "--db-type",
        default="sqlite",
        choices=["sqlite", "postgresql"],
        help="Database type",
    )
    parser.add_argument(
        "--db-path", default="news_market_predictor.db", help="SQLite database path"
    )
    parser.add_argument(
        "--use-env", action="store_true", help="Use environment variables for config"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init command
    subparsers.add_parser("init", help="Initialize database with schema")

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run database migrations")
    migrate_parser.add_argument(
        "--direction",
        choices=["up", "down"],
        default="up",
        help="Migration direction",
    )
    migrate_parser.add_argument(
        "--version",
        type=int,
        help="Target version (default: latest for up, 0 for down)",
    )

    # Status command
    subparsers.add_parser("status", help="Show migration status")

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create database backup")
    backup_parser.add_argument(
        "--backup-dir", default="backups", help="Backup directory"
    )
    backup_parser.add_argument(
        "--compress", action="store_true", default=True, help="Compress backup"
    )
    backup_parser.add_argument(
        "--verify", action="store_true", default=True, help="Verify backup"
    )

    # List backups command
    list_parser = subparsers.add_parser("list-backups", help="List available backups")
    list_parser.add_argument("--backup-dir", default="backups", help="Backup directory")

    # Restore command
    restore_parser = subparsers.add_parser(
        "restore", help="Restore database from backup"
    )
    restore_parser.add_argument("backup_file", help="Backup file to restore")
    restore_parser.add_argument(
        "--backup-dir", default="backups", help="Backup directory"
    )

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    commands = {
        "init": init_database,
        "migrate": migrate_database,
        "status": migration_status,
        "backup": create_backup,
        "list-backups": list_backups,
        "restore": restore_backup,
        "stats": database_stats,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
