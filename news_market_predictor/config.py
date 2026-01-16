"""
Configuration management for the News Market Predictor system.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .exceptions import ConfigurationError


@dataclass
class NetworkConfig:
    """Network-related configuration."""

    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    rate_limit_delay: float = 0.1


@dataclass
class ProcessingConfig:
    """Content processing configuration."""

    batch_size: int = 10
    max_content_length: int = 50000
    min_confidence_threshold: float = 0.3


@dataclass
class StorageConfig:
    """Data storage configuration."""

    database_url: Optional[str] = None
    backup_enabled: bool = True
    data_retention_days: int = 90


@dataclass
class Config:
    """Main configuration class."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    log_level: str = "INFO"
    log_file: str = "news_predictor.log"

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        config = cls()

        # Network configuration
        if os.getenv("MAX_RETRIES"):
            config.network.max_retries = int(os.getenv("MAX_RETRIES"))
        if os.getenv("RETRY_DELAY"):
            config.network.retry_delay = float(os.getenv("RETRY_DELAY"))
        if os.getenv("TIMEOUT"):
            config.network.timeout = int(os.getenv("TIMEOUT"))
        if os.getenv("RATE_LIMIT_DELAY"):
            config.network.rate_limit_delay = float(os.getenv("RATE_LIMIT_DELAY"))

        # Processing configuration
        if os.getenv("BATCH_SIZE"):
            config.processing.batch_size = int(os.getenv("BATCH_SIZE"))
        if os.getenv("MAX_CONTENT_LENGTH"):
            config.processing.max_content_length = int(os.getenv("MAX_CONTENT_LENGTH"))
        if os.getenv("MIN_CONFIDENCE_THRESHOLD"):
            config.processing.min_confidence_threshold = float(
                os.getenv("MIN_CONFIDENCE_THRESHOLD")
            )

        # Storage configuration
        config.storage.database_url = os.getenv("DATABASE_URL")
        if os.getenv("BACKUP_ENABLED"):
            config.storage.backup_enabled = (
                os.getenv("BACKUP_ENABLED").lower() == "true"
            )
        if os.getenv("DATA_RETENTION_DAYS"):
            config.storage.data_retention_days = int(os.getenv("DATA_RETENTION_DAYS"))

        # Logging configuration
        config.log_level = os.getenv("LOG_LEVEL", "INFO")
        config.log_file = os.getenv("LOG_FILE", "news_predictor.log")

        return config

    def validate(self) -> None:
        """Validate configuration values."""
        if self.network.max_retries < 0:
            raise ConfigurationError("max_retries must be non-negative")

        if self.network.retry_delay < 0:
            raise ConfigurationError("retry_delay must be non-negative")

        if self.network.timeout <= 0:
            raise ConfigurationError("timeout must be positive")

        if not 0 <= self.processing.min_confidence_threshold <= 1:
            raise ConfigurationError("min_confidence_threshold must be between 0 and 1")

        if self.processing.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")

        if self.storage.data_retention_days < 0:
            raise ConfigurationError("data_retention_days must be non-negative")
