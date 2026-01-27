"""
Tests for Google Gemini LLM provider.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from news_market_predictor.llm.providers.gemini_provider import (
    GeminiProvider,
    GeminiConfig,
)
from news_market_predictor.llm.models import LLMResponse
from news_market_predictor.llm.exceptions import (
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMInvalidResponseError,
)


class TestGeminiConfig:
    """Test Gemini configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GeminiConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.model == "gemini-1.5-flash"
        assert config.max_requests_per_minute == 15
        assert config.max_requests_per_day == 1500
        assert config.cost_per_1k_input_tokens == 0.0  # Free tier


class TestGeminiProvider:
    """Test Gemini provider implementation."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        config = GeminiConfig(api_key="test-key")
        provider = GeminiProvider(config)

        assert provider.get_provider_name() == "gemini"
        assert provider.config.api_key == "test-key"

    def test_provider_initialization_no_api_key(self):
        """Test provider initialization without API key."""
        config = GeminiConfig(api_key="")

        with pytest.raises(LLMProviderError) as exc_info:
            GeminiProvider(config)

        assert "API key is required" in str(exc_info.value)

    def test_is_available_with_valid_key(self):
        """Test availability check with valid API key."""
        config = GeminiConfig(api_key="valid-test-key-12345")
        provider = GeminiProvider(config)

        # Should return True for valid-looking API key
        assert provider.is_available() is True

    def test_is_available_with_invalid_key(self):
        """Test availability check with invalid API key."""
        config = GeminiConfig(api_key="short")
        provider = GeminiProvider(config)

        # Should return False for short API key
        assert provider.is_available() is False

    @patch("requests.post")
    def test_generate_completion_success(self, mock_post):
        """Test successful completion generation."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "This is a test response from Gemini."}]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        config = GeminiConfig(api_key="test-key")
        provider = GeminiProvider(config)

        response = provider.generate_completion("Test prompt")

        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response from Gemini."
        assert response.provider == "gemini"
        assert response.model_used == "gemini-1.5-flash"
        assert response.tokens_used > 0
        assert response.cost_estimate == 0.0  # Free tier

    @patch("requests.post")
    def test_generate_completion_rate_limit(self, mock_post):
        """Test rate limit handling."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_post.return_value = mock_response

        config = GeminiConfig(api_key="test-key")
        provider = GeminiProvider(config)

        with pytest.raises(LLMRateLimitError) as exc_info:
            provider.generate_completion("Test prompt")

        assert "rate limit exceeded" in str(exc_info.value).lower()

    @patch("requests.post")
    def test_generate_completion_timeout(self, mock_post):
        """Test timeout handling."""
        # Mock timeout
        mock_post.side_effect = Exception("timeout")

        config = GeminiConfig(api_key="test-key", timeout_seconds=1)
        provider = GeminiProvider(config)

        with pytest.raises(LLMProviderError):
            provider.generate_completion("Test prompt")

    @patch("requests.post")
    def test_generate_structured_output_success(self, mock_post):
        """Test successful structured output generation."""
        # Mock successful API response with JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"sentiment": "positive", "confidence": 0.8}'}
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        config = GeminiConfig(api_key="test-key")
        provider = GeminiProvider(config)

        schema = {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["sentiment", "confidence"],
        }

        result = provider.generate_structured_output("Analyze sentiment", schema)

        assert isinstance(result, dict)
        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.8

    @patch("requests.post")
    def test_generate_structured_output_invalid_json(self, mock_post):
        """Test handling of invalid JSON in structured output."""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "This is not valid JSON"}]}}]
        }
        mock_post.return_value = mock_response

        config = GeminiConfig(api_key="test-key")
        provider = GeminiProvider(config)

        schema = {"type": "object", "properties": {"sentiment": {"type": "string"}}}

        with pytest.raises(LLMInvalidResponseError) as exc_info:
            provider.generate_structured_output("Analyze sentiment", schema)

        assert "Invalid JSON response" in str(exc_info.value)

    def test_estimate_cost_free_tier(self):
        """Test cost estimation for free tier."""
        config = GeminiConfig(api_key="test-key")
        provider = GeminiProvider(config)

        cost = provider.estimate_cost("Test prompt", max_tokens=100)
        assert cost == 0.0  # Free tier

    def test_rate_limiting_logic(self):
        """Test rate limiting logic."""
        config = GeminiConfig(api_key="test-key")
        provider = GeminiProvider(config)

        # Initially should be able to make requests
        assert provider._can_make_request() is True

        # Simulate hitting daily limit
        provider.daily_request_count = config.max_requests_per_day
        assert provider._can_make_request() is False

        # Reset and test minute limit
        provider.daily_request_count = 0

        # Simulate hitting per-minute limit
        import time

        now = time.time()
        provider.request_timestamps = [now - 30] * config.max_requests_per_minute
        assert provider._can_make_request() is False

    def test_usage_stats_tracking(self):
        """Test usage statistics tracking."""
        config = GeminiConfig(api_key="test-key")
        provider = GeminiProvider(config)

        # Initial stats
        stats = provider.get_usage_stats()
        assert stats.provider == "gemini"
        assert stats.total_requests == 0

        # Simulate successful request
        mock_response = LLMResponse(
            content="test",
            model_used="gemini-1.5-flash",
            provider="gemini",
            tokens_used=10,
            cost_estimate=0.0,
            confidence_score=0.8,
        )
        provider._record_successful_request(mock_response, 1.0)

        # Check updated stats
        stats = provider.get_usage_stats()
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.total_tokens_used == 10

    def test_reset_usage_stats(self):
        """Test usage statistics reset."""
        config = GeminiConfig(api_key="test-key")
        provider = GeminiProvider(config)

        # Simulate some usage
        provider.daily_request_count = 10
        provider.request_timestamps = [1234567890.0]

        # Reset stats
        provider.reset_usage_stats()

        # Check reset
        assert provider.daily_request_count == 0
        assert len(provider.request_timestamps) == 0

        stats = provider.get_usage_stats()
        assert stats.total_requests == 0
