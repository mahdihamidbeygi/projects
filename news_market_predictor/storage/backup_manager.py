"""
Backup and recovery manager for database operations.
Handles automated backups, recovery, and backup rotation.
"""

import os
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Configuration for backup operations."""

    backup_dir: str = "backups"
    max_backups: int = 7  # Keep last 7 backups
    backup_frequency_hours: int = 24  # Daily backups
    compress_backups: bool = True
    verify_backups: bool = True


class BackupManager:
    """Manages database backups and recovery operations."""

    def __init__(self, db_path: str, config: Optional[BackupConfig] = None):
        """
        Initialize backup manager.

        Args:
            db_path: Path to the database file
            config: Backup configuration
        """
        self.db_path = db_path
        self.config = config or BackupConfig()
        self._ensure_backup_directory()
        self._last_backup_time: Optional[datetime] = None

    def _ensure_backup_directory(self) -> None:
        """Ensure backup directory exists."""
        backup_path = Path(self.config.backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backup directory ensured at {self.config.backup_dir}")

    def should_create_backup(self) -> bool:
        """
        Check if a backup should be created based on frequency policy.

        Returns:
            True if backup should be created, False otherwise
        """
        if self._last_backup_time is None:
            return True

        time_since_backup = datetime.now() - self._last_backup_time
        return time_since_backup.total_seconds() >= (
            self.config.backup_frequency_hours * 3600
        )

    def create_backup(self, force: bool = False) -> Dict[str, Any]:
        """
        Create a database backup.

        Args:
            force: Force backup even if frequency policy says not to

        Returns:
            Dictionary with backup results
        """
        if not force and not self.should_create_backup():
            return {
                "status": "skipped",
                "reason": "Backup frequency policy not met",
                "last_backup": (
                    self._last_backup_time.isoformat()
                    if self._last_backup_time
                    else None
                ),
            }

        logger.info("Creating database backup")
        backup_result = {
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "db_path": self.db_path,
        }

        try:
            # Check if database file exists
            if not Path(self.db_path).exists():
                backup_result["status"] = "failed"
                backup_result["error"] = "Database file does not exist"
                logger.error(f"Database file not found: {self.db_path}")
                return backup_result

            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"backup_{timestamp}.db"
            backup_path = Path(self.config.backup_dir) / backup_filename

            # Copy database file
            shutil.copy2(self.db_path, backup_path)
            backup_result["backup_file"] = str(backup_path)
            backup_result["backup_size_bytes"] = backup_path.stat().st_size

            # Compress if configured
            if self.config.compress_backups:
                compressed_path = self._compress_backup(backup_path)
                if compressed_path:
                    backup_result["compressed_file"] = str(compressed_path)
                    backup_result["compressed_size_bytes"] = (
                        compressed_path.stat().st_size
                    )
                    # Remove uncompressed file
                    backup_path.unlink()
                    backup_path = compressed_path

            # Verify backup if configured
            if self.config.verify_backups:
                verification = self._verify_backup(backup_path)
                backup_result["verification"] = verification
                if not verification["valid"]:
                    backup_result["status"] = "failed"
                    backup_result["error"] = "Backup verification failed"
                    logger.error("Backup verification failed")
                    return backup_result

            # Rotate old backups
            rotation_result = self._rotate_backups()
            backup_result["rotation"] = rotation_result

            # Update status
            backup_result["status"] = "completed"
            backup_result["end_time"] = datetime.now().isoformat()
            self._last_backup_time = datetime.now()

            logger.info(f"Backup created successfully: {backup_path}")

        except Exception as e:
            backup_result["status"] = "failed"
            backup_result["error"] = str(e)
            backup_result["end_time"] = datetime.now().isoformat()
            logger.error(f"Backup creation failed: {e}")

        return backup_result

    def _compress_backup(self, backup_path: Path) -> Optional[Path]:
        """
        Compress a backup file using gzip.

        Args:
            backup_path: Path to backup file

        Returns:
            Path to compressed file or None if compression failed
        """
        try:
            import gzip

            compressed_path = backup_path.with_suffix(backup_path.suffix + ".gz")

            with open(backup_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            logger.debug(f"Compressed backup to {compressed_path}")
            return compressed_path

        except Exception as e:
            logger.error(f"Failed to compress backup: {e}")
            return None

    def _verify_backup(self, backup_path: Path) -> Dict[str, Any]:
        """
        Verify backup file integrity.

        Args:
            backup_path: Path to backup file

        Returns:
            Dictionary with verification results
        """
        verification = {"valid": False, "checks": []}

        try:
            # Check file exists
            if not backup_path.exists():
                verification["checks"].append({"check": "file_exists", "passed": False})
                return verification

            verification["checks"].append({"check": "file_exists", "passed": True})

            # Check file size
            file_size = backup_path.stat().st_size
            if file_size == 0:
                verification["checks"].append({"check": "file_size", "passed": False})
                return verification

            verification["checks"].append(
                {"check": "file_size", "passed": True, "size_bytes": file_size}
            )

            # For compressed files, try to decompress
            if backup_path.suffix == ".gz":
                import gzip

                try:
                    with gzip.open(backup_path, "rb") as f:
                        # Read first few bytes to verify it's valid gzip
                        f.read(1024)
                    verification["checks"].append(
                        {"check": "gzip_valid", "passed": True}
                    )
                except Exception:
                    verification["checks"].append(
                        {"check": "gzip_valid", "passed": False}
                    )
                    return verification

            # For SQLite files, try to open
            elif backup_path.suffix == ".db":
                import sqlite3

                try:
                    conn = sqlite3.connect(str(backup_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    conn.close()

                    verification["checks"].append(
                        {
                            "check": "sqlite_valid",
                            "passed": True,
                            "table_count": len(tables),
                        }
                    )
                except Exception:
                    verification["checks"].append(
                        {"check": "sqlite_valid", "passed": False}
                    )
                    return verification

            # All checks passed
            verification["valid"] = True

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            verification["error"] = str(e)

        return verification

    def _rotate_backups(self) -> Dict[str, Any]:
        """
        Rotate old backups based on retention policy.

        Returns:
            Dictionary with rotation results
        """
        rotation_result = {"removed_backups": [], "kept_backups": []}

        try:
            backup_dir = Path(self.config.backup_dir)
            backup_files = sorted(
                backup_dir.glob("backup_*.db*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            # Keep only max_backups most recent
            for i, backup_file in enumerate(backup_files):
                if i < self.config.max_backups:
                    rotation_result["kept_backups"].append(str(backup_file))
                else:
                    backup_file.unlink()
                    rotation_result["removed_backups"].append(str(backup_file))
                    logger.debug(f"Removed old backup: {backup_file}")

        except Exception as e:
            logger.error(f"Backup rotation failed: {e}")
            rotation_result["error"] = str(e)

        return rotation_result

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.

        Returns:
            List of backup information dictionaries
        """
        backups = []

        try:
            backup_dir = Path(self.config.backup_dir)
            backup_files = sorted(
                backup_dir.glob("backup_*.db*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            for backup_file in backup_files:
                stat = backup_file.stat()
                backups.append(
                    {
                        "filename": backup_file.name,
                        "path": str(backup_file),
                        "size_bytes": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "compressed": backup_file.suffix == ".gz",
                    }
                )

        except Exception as e:
            logger.error(f"Failed to list backups: {e}")

        return backups

    def restore_backup(self, backup_filename: str) -> Dict[str, Any]:
        """
        Restore database from a backup.

        Args:
            backup_filename: Name of backup file to restore

        Returns:
            Dictionary with restore results
        """
        restore_result = {
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "backup_file": backup_filename,
        }

        try:
            backup_path = Path(self.config.backup_dir) / backup_filename

            # Check if backup exists
            if not backup_path.exists():
                restore_result["status"] = "failed"
                restore_result["error"] = "Backup file not found"
                logger.error(f"Backup file not found: {backup_path}")
                return restore_result

            # Create a backup of current database before restoring
            current_backup = self.create_backup(force=True)
            restore_result["current_db_backup"] = current_backup

            # Decompress if needed
            restore_path = backup_path
            if backup_path.suffix == ".gz":
                restore_path = self._decompress_backup(backup_path)
                if not restore_path:
                    restore_result["status"] = "failed"
                    restore_result["error"] = "Failed to decompress backup"
                    return restore_result

            # Restore database
            shutil.copy2(restore_path, self.db_path)

            # Clean up decompressed file if it was temporary
            if restore_path != backup_path:
                restore_path.unlink()

            restore_result["status"] = "completed"
            restore_result["end_time"] = datetime.now().isoformat()
            logger.info(f"Database restored from backup: {backup_filename}")

        except Exception as e:
            restore_result["status"] = "failed"
            restore_result["error"] = str(e)
            restore_result["end_time"] = datetime.now().isoformat()
            logger.error(f"Database restore failed: {e}")

        return restore_result

    def _decompress_backup(self, compressed_path: Path) -> Optional[Path]:
        """
        Decompress a backup file.

        Args:
            compressed_path: Path to compressed backup

        Returns:
            Path to decompressed file or None if decompression failed
        """
        try:
            import gzip

            decompressed_path = compressed_path.with_suffix("")

            with gzip.open(compressed_path, "rb") as f_in:
                with open(decompressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            logger.debug(f"Decompressed backup to {decompressed_path}")
            return decompressed_path

        except Exception as e:
            logger.error(f"Failed to decompress backup: {e}")
            return None

    def get_backup_status(self) -> Dict[str, Any]:
        """
        Get current backup status.

        Returns:
            Dictionary with backup status information
        """
        backups = self.list_backups()

        status = {
            "backup_dir": self.config.backup_dir,
            "total_backups": len(backups),
            "max_backups": self.config.max_backups,
            "last_backup": (
                self._last_backup_time.isoformat() if self._last_backup_time else None
            ),
            "should_create_backup": self.should_create_backup(),
            "backups": backups,
        }

        if backups:
            total_size = sum(b["size_bytes"] for b in backups)
            status["total_backup_size_bytes"] = total_size
            status["oldest_backup"] = backups[-1]["created_at"]
            status["newest_backup"] = backups[0]["created_at"]

        return status
