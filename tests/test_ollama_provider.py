"""
Tests for Ollama LLM provider.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from news_market_predictor.llm.providers.ollama_provider import (
    OllamaProvider,
    OllamaConfig,
)
from news_market_predictor.llm.models import LLMResponse
from news_market_predictor.llm.exceptions import (
    LLMProviderError,
    LLMTimeoutError,
    LLMInvalidResponseError,
)


class TestOllamaConfig:
    """Test Ollama configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OllamaConfig()

        assert config.base_url == "http://localhost:11434"
        assert config.model == "llama3.2"
        assert config.timeout_seconds == 60
        assert "llama3.2" in config.available_models
        assert "mistral" in config.available_models

    def test_custom_config(self):
        """Test custom configuration."""
        config = OllamaConfig(
            base_url="http://custom:8080", model="custom-model", timeout_seconds=30
        )

        assert config.base_url == "http://custom:8080"
        assert config.model == "custom-model"
        assert config.timeout_seconds == 30


class TestOllamaProvider:
    """Test Ollama provider implementation."""

    @patch("requests.get")
    def test_provider_initialization(self, mock_get):
        """Test provider initialization."""
        # Mock successful connection check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        config = OllamaConfig()
        provider = OllamaProvider(config)

        assert provider.get_provider_name() == "ollama"
        assert provider.config.base_url == "http://localhost:11434"

    @patch("requests.get")
    def test_is_available_success(self, mock_get):
        """Test availability check when Ollama is running."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        config = OllamaConfig()
        provider = OllamaProvider(config)

        assert provider.is_available() is True

    @patch("requests.get")
    def test_is_available_failure(self, mock_get):
        """Test availability check when Ollama is not running."""
        # Mock connection error
        mock_get.side_effect = Exception("Connection refused")

        config = OllamaConfig()
        provider = OllamaProvider(config)

        assert provider.is_available() is False

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_completion_success(self, mock_post, mock_get):
        """Test successful completion generation."""
        # Mock connection check
        mock_get.return_value = Mock(status_code=200)

        # Mock model availability check
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

        # Mock successful completion response
        mock_completion_response = Mock()
        mock_completion_response.status_code = 200
        mock_completion_response.json.return_value = {
            "response": "This is a test response from Ollama.",
            "eval_count": 15,
            "prompt_eval_count": 10,
            "total_duration": 1000000000,  # 1 second in nanoseconds
            "eval_duration": 500000000,
            "prompt_eval_duration": 300000000,
            "load_duration": 200000000,
        }
        mock_post.return_value = mock_completion_response

        config = OllamaConfig()
        provider = OllamaProvider(config)

        response = provider.generate_completion("Test prompt")

        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response from Ollama."
        assert response.provider == "ollama"
        assert response.model_used == "llama3.2"
        assert response.tokens_used == 25  # prompt_eval_count + eval_count
        assert response.cost_estimate == 0.0  # Local inference is free

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_completion_model_not_available(self, mock_post, mock_get):
        """Test completion when model is not available."""
        # Mock connection check
        mock_get.return_value = Mock(status_code=200)

        # Mock model not available
        mock_get.return_value.json.return_value = {"models": []}

        # Mock model pull failure
        mock_pull_response = Mock()
        mock_pull_response.status_code = 404
        mock_post.return_value = mock_pull_response

        config = OllamaConfig()
        provider = OllamaProvider(config)

        with pytest.raises(LLMProviderError) as exc_info:
            provider.generate_completion("Test prompt")

        assert "not available and could not be pulled" in str(exc_info.value)

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_completion_timeout(self, mock_post, mock_get):
        """Test timeout handling."""
        # Mock connection check
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

        # Mock timeout
        mock_post.side_effect = Exception("timeout")

        config = OllamaConfig(timeout_seconds=1)
        provider = OllamaProvider(config)

        with pytest.raises(LLMProviderError):
            provider.generate_completion("Test prompt")

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_structured_output_success(self, mock_post, mock_get):
        """Test successful structured output generation."""
        # Mock connection check
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

        # Mock successful response with JSON
        mock_completion_response = Mock()
        mock_completion_response.status_code = 200
        mock_completion_response.json.return_value = {
            "response": '{"sentiment": "positive", "confidence": 0.8}',
            "eval_count": 10,
            "prompt_eval_count": 5,
        }
        mock_post.return_value = mock_completion_response

        config = OllamaConfig()
        provider = OllamaProvider(config)

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

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_structured_output_with_extra_text(self, mock_post, mock_get):
        """Test structured output with extra text around JSON."""
        # Mock connection check
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

        # Mock response with JSON embedded in text
        mock_completion_response = Mock()
        mock_completion_response.status_code = 200
        mock_completion_response.json.return_value = {
            "response": 'Here is the analysis: {"sentiment": "negative", "confidence": 0.9} Hope this helps!',
            "eval_count": 10,
            "prompt_eval_count": 5,
        }
        mock_post.return_value = mock_completion_response

        config = OllamaConfig()
        provider = OllamaProvider(config)

        schema = {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }

        result = provider.generate_structured_output("Analyze sentiment", schema)

        assert isinstance(result, dict)
        assert result["sentiment"] == "negative"
        assert result["confidence"] == 0.9

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_structured_output_invalid_json(self, mock_post, mock_get):
        """Test handling of invalid JSON in structured output."""
        # Mock connection check
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

        # Mock response with invalid JSON
        mock_completion_response = Mock()
        mock_completion_response.status_code = 200
        mock_completion_response.json.return_value = {
            "response": "This is not valid JSON at all",
            "eval_count": 10,
            "prompt_eval_count": 5,
        }
        mock_post.return_value = mock_completion_response

        config = OllamaConfig()
        provider = OllamaProvider(config)

        schema = {"type": "object", "properties": {"sentiment": {"type": "string"}}}

        with pytest.raises(LLMInvalidResponseError) as exc_info:
            provider.generate_structured_output("Analyze sentiment", schema)

        assert "Invalid JSON response" in str(exc_info.value)

    def test_estimate_cost_always_free(self):
        """Test cost estimation is always free for local inference."""
        config = OllamaConfig()
        provider = OllamaProvider(config)

        cost = provider.estimate_cost("Test prompt", max_tokens=1000)
        assert cost == 0.0  # Local inference is always free

    @patch("requests.get")
    def test_list_available_models_success(self, mock_get):
        """Test listing available models."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2"},
                {"name": "mistral:7b"},
                {"name": "codellama"},
            ]
        }
        mock_get.return_value = mock_response

        config = OllamaConfig()
        provider = OllamaProvider(config)

        models = provider.list_available_models()
        assert "llama3.2" in models
        assert "mistral:7b" in models
        assert "codellama" in models

    @patch("requests.get")
    def test_list_available_models_failure(self, mock_get):
        """Test listing models when request fails."""
        # Mock failed response
        mock_get.side_effect = Exception("Connection error")

        config = OllamaConfig()
        provider = OllamaProvider(config)

        models = provider.list_available_models()
        assert models == []

    @patch("requests.get")
    def test_is_model_available(self, mock_get):
        """Test model availability checking."""
        # Mock available models
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2"}, {"name": "mistral:7b"}]
        }
        mock_get.return_value = mock_response

        config = OllamaConfig()
        provider = OllamaProvider(config)

        # Test exact match
        assert provider._is_model_available("llama3.2") is True

        # Test tag match
        assert provider._is_model_available("mistral") is True

        # Test not available
        assert provider._is_model_available("nonexistent") is False

    def test_usage_stats_tracking(self):
        """Test usage statistics tracking."""
        config = OllamaConfig()
        provider = OllamaProvider(config)

        # Initial stats
        stats = provider.get_usage_stats()
        assert stats.provider == "ollama"
        assert stats.total_requests == 0

        # Simulate successful request
        mock_response = LLMResponse(
            content="test",
            model_used="llama3.2",
            provider="ollama",
            tokens_used=20,
            cost_estimate=0.0,
            confidence_score=0.7,
        )
        provider._record_successful_request(mock_response, 2.0)

        # Check updated stats
        stats = provider.get_usage_stats()
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.total_tokens_used == 20
        assert stats.average_response_time == 2.0

    def test_reset_usage_stats(self):
        """Test usage statistics reset."""
        config = OllamaConfig()
        provider = OllamaProvider(config)

        # Simulate some usage
        provider.usage_stats.total_requests = 10
        provider.usage_stats.successful_requests = 8

        # Reset stats
        provider.reset_usage_stats()

        # Check reset
        stats = provider.get_usage_stats()
        assert stats.total_requests == 0
        assert stats.successful_requests == 0

    def test_token_calculation_from_response(self):
        """Test token calculation from Ollama response."""
        config = OllamaConfig()
        provider = OllamaProvider(config)

        # Test with token counts in response
        response_with_tokens = {
            "response": "test response",
            "prompt_eval_count": 10,
            "eval_count": 15,
        }
        tokens = provider._calculate_tokens(response_with_tokens)
        assert tokens == 25  # 10 + 15

        # Test without token counts (fallback to estimation)
        response_without_tokens = {
            "response": "this is a test response with multiple words"
        }
        tokens = provider._calculate_tokens(response_without_tokens)
        assert tokens > 0  # Should estimate based on word count
