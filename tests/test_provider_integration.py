"""
Integration tests for LLM providers with service manager.
"""

import pytest
from unittest.mock import Mock, patch

from news_market_predictor.llm.service_manager import LLMServiceManager
from news_market_predictor.llm.models import LLMConfiguration
from news_market_predictor.llm.providers.gemini_provider import (
    GeminiProvider,
    GeminiConfig,
)
from news_market_predictor.llm.providers.ollama_provider import (
    OllamaProvider,
    OllamaConfig,
)


class TestProviderIntegration:
    """Test integration of providers with service manager."""

    @patch("requests.get")
    @patch("requests.post")
    def test_gemini_provider_integration(self, mock_post, mock_get):
        """Test Gemini provider integration with service manager."""
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

        # Create service manager
        config = LLMConfiguration(default_provider="gemini")
        manager = LLMServiceManager(config)

        # Add Gemini provider
        gemini_config = GeminiConfig(api_key="test-key")
        gemini_provider = GeminiProvider(gemini_config)
        manager.add_provider(gemini_provider)

        # Test completion generation
        response = manager.generate_completion("Test prompt")

        assert response.provider == "gemini"
        assert response.content == "This is a test response from Gemini."
        assert response.cost_estimate == 0.0  # Free tier

    @patch("requests.get")
    @patch("requests.post")
    def test_ollama_provider_integration(self, mock_post, mock_get):
        """Test Ollama provider integration with service manager."""
        # Mock connection check
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

        # Mock successful completion response
        mock_completion_response = Mock()
        mock_completion_response.status_code = 200
        mock_completion_response.json.return_value = {
            "response": "This is a test response from Ollama.",
            "eval_count": 15,
            "prompt_eval_count": 10,
        }
        mock_post.return_value = mock_completion_response

        # Create service manager
        config = LLMConfiguration(default_provider="ollama")
        manager = LLMServiceManager(config)

        # Add Ollama provider
        ollama_config = OllamaConfig()
        ollama_provider = OllamaProvider(ollama_config)
        manager.add_provider(ollama_provider)

        # Test completion generation
        response = manager.generate_completion("Test prompt")

        assert response.provider == "ollama"
        assert response.content == "This is a test response from Ollama."
        assert response.cost_estimate == 0.0  # Local inference is free

    @patch("requests.get")
    @patch("requests.post")
    def test_provider_fallback_gemini_to_ollama(self, mock_post, mock_get):
        """Test fallback from Gemini to Ollama when Gemini fails."""
        # Mock Ollama connection check
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

        # Mock Gemini failure and Ollama success
        def mock_post_side_effect(url, **kwargs):
            if "generativelanguage.googleapis.com" in url:
                # Gemini fails with rate limit
                mock_response = Mock()
                mock_response.status_code = 429
                mock_response.headers = {"Retry-After": "60"}
                return mock_response
            else:
                # Ollama succeeds
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "response": "Fallback response from Ollama.",
                    "eval_count": 10,
                    "prompt_eval_count": 5,
                }
                return mock_response

        mock_post.side_effect = mock_post_side_effect

        # Create service manager with fallback enabled
        config = LLMConfiguration(default_provider="gemini", fallback_enabled=True)
        manager = LLMServiceManager(config)

        # Add both providers
        gemini_config = GeminiConfig(api_key="test-key")
        gemini_provider = GeminiProvider(gemini_config)
        manager.add_provider(gemini_provider)

        ollama_config = OllamaConfig()
        ollama_provider = OllamaProvider(ollama_config)
        manager.add_provider(ollama_provider)

        # Test completion generation - should fallback to Ollama
        response = manager.generate_completion("Test prompt")

        assert response.provider == "ollama"
        assert response.content == "Fallback response from Ollama."

    @patch("requests.get")
    @patch("requests.post")
    def test_structured_output_integration(self, mock_post, mock_get):
        """Test structured output generation with both providers."""
        # Mock Gemini response
        mock_gemini_response = Mock()
        mock_gemini_response.status_code = 200
        mock_gemini_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"sentiment": "positive", "confidence": 0.9}'}
                        ]
                    }
                }
            ]
        }

        # Mock Ollama connection check
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

        # Mock Ollama response
        mock_ollama_response = Mock()
        mock_ollama_response.status_code = 200
        mock_ollama_response.json.return_value = {
            "response": '{"sentiment": "negative", "confidence": 0.8}',
            "eval_count": 8,
            "prompt_eval_count": 12,
        }

        def mock_post_side_effect(url, **kwargs):
            if "generativelanguage.googleapis.com" in url:
                return mock_gemini_response
            else:
                return mock_ollama_response

        mock_post.side_effect = mock_post_side_effect

        # Create service manager
        config = LLMConfiguration()
        manager = LLMServiceManager(config)

        # Add both providers
        gemini_config = GeminiConfig(api_key="test-key")
        gemini_provider = GeminiProvider(gemini_config)
        manager.add_provider(gemini_provider)

        ollama_config = OllamaConfig()
        ollama_provider = OllamaProvider(ollama_config)
        manager.add_provider(ollama_provider)

        schema = {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["sentiment", "confidence"],
        }

        # Test with Gemini
        result_gemini = manager.generate_structured_output(
            "Analyze sentiment", schema, provider="gemini"
        )
        assert result_gemini["sentiment"] == "positive"
        assert result_gemini["confidence"] == 0.9

        # Test with Ollama
        result_ollama = manager.generate_structured_output(
            "Analyze sentiment", schema, provider="ollama"
        )
        assert result_ollama["sentiment"] == "negative"
        assert result_ollama["confidence"] == 0.8

    def test_provider_usage_stats_integration(self):
        """Test usage statistics collection across providers."""
        # Create service manager
        config = LLMConfiguration()
        manager = LLMServiceManager(config)

        # Add providers (without making actual requests)
        gemini_config = GeminiConfig(api_key="test-key")
        gemini_provider = GeminiProvider(gemini_config)
        manager.add_provider(gemini_provider)

        ollama_config = OllamaConfig()
        ollama_provider = OllamaProvider(ollama_config)
        manager.add_provider(ollama_provider)

        # Check initial stats
        stats = manager.get_usage_stats()
        assert "gemini" in stats
        assert "ollama" in stats
        assert stats["gemini"].total_requests == 0
        assert stats["ollama"].total_requests == 0

        # Check individual provider stats
        gemini_stats = manager.get_usage_stats("gemini")
        assert "gemini" in gemini_stats
        assert len(gemini_stats) == 1
