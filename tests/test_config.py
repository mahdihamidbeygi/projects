"""
Tests for configuration management.
"""

import pytest
import os

from news_market_predictor.config import (
    Config,
    NetworkConfig,
    ProcessingConfig,
    StorageConfig,
)
from news_market_predictor.exceptions import ConfigurationError


def test_default_config():
    """Test default configuration creation."""
    config = Config()

    assert config.network.max_retries == 3
    assert config.processing.batch_size == 10
    assert config.log_level == "INFO"


def test_config_validation_valid():
    """Test configuration validation with valid values."""
    config = Config()
    config.validate()  # Should not raise


def test_config_validation_invalid_retries():
    """Test configuration validation with invalid retry count."""
    config = Config()
    config.network.max_retries = -1

    with pytest.raises(ConfigurationError):
        config.validate()


def test_config_validation_invalid_confidence():
    """Test configuration validation with invalid confidence threshold."""
    config = Config()
    config.processing.min_confidence_threshold = 1.5

    with pytest.raises(ConfigurationError):
        config.validate()


def test_config_from_env(monkeypatch):
    """Test configuration creation from environment variables."""
    monkeypatch.setenv("MAX_RETRIES", "5")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("BATCH_SIZE", "20")

    config = Config.from_env()

    assert config.network.max_retries == 5
    assert config.log_level == "DEBUG"
    assert config.processing.batch_size == 20
