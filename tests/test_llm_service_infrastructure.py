"""
Tests for LLM service infrastructure and interfaces.
"""

import pytest
from datetime import datetime

from news_market_predictor.llm.models import (
    LLMResponse,
    LLMConfiguration,
    LLMUsageStats,
)
from news_market_predictor.llm.service_manager import LLMServiceManager
from news_market_predictor.llm.providers.mock_provider import MockLLMProvider
from news_market_predictor.llm.exceptions import (
    LLMServiceError,
    LLMProviderError,
    LLMConfigurationError,
    LLMBudgetExceededError,
)


class TestLLMModels:
    """Test LLM data models."""

    def test_llm_response_validation(self):
        """Test LLMResponse validation."""
        response = LLMResponse(
            content="Test response",
            model_used="test-model",
            provider="test-provider",
            tokens_used=100,
            cost_estimate=0.01,
            confidence_score=0.8,
            reasoning_chain=["step1", "step2"],
            metadata={"key": "value"},
        )

        assert response.validate() is True

        # Test serialization
        json_str = response.to_json()
        assert isinstance(json_str, str)

        # Test deserialization
        restored = LLMResponse.from_json(json_str)
        assert restored.content == response.content
        assert restored.model_used == response.model_used
        assert restored.provider == response.provider

    def test_llm_configuration_validation(self):
        """Test LLMConfiguration validation."""
        config = LLMConfiguration(
            enabled=True,
            default_provider="openai",
            model_name="gpt-4",
            max_tokens=2000,
            temperature=0.1,
            timeout_seconds=30,
            cost_limit_per_hour=10.0,
        )

        assert config.validate() is True

        # Test serialization
        json_str = config.to_json()
        assert isinstance(json_str, str)

        # Test deserialization
        restored = LLMConfiguration.from_json(json_str)
        assert restored.enabled == config.enabled
        assert restored.default_provider == config.default_provider

    def test_llm_usage_stats_validation(self):
        """Test LLMUsageStats validation."""
        stats = LLMUsageStats(
            provider="test-provider",
            total_requests=10,
            successful_requests=8,
            failed_requests=2,
            total_tokens_used=1000,
            total_cost=0.1,
            average_response_time=1.5,
        )

        assert stats.validate() is True

        # Test serialization
        json_str = stats.to_json()
        assert isinstance(json_str, str)


class TestMockLLMProvider:
    """Test mock LLM provider."""

    def test_provider_basic_functionality(self):
        """Test basic provider functionality."""
        provider = MockLLMProvider("test-mock")

        assert provider.get_provider_name() == "test-mock"
        assert provider.is_available() is True

        # Test completion generation
        response = provider.generate_completion("Test prompt")
        assert isinstance(response, LLMResponse)
        assert response.provider == "test-mock"
        assert response.tokens_used > 0
        assert response.cost_estimate > 0

        # Test structured output
        schema = {"sentiment": "string", "confidence": "number"}
        result = provider.generate_structured_output("Test prompt", schema)
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "confidence" in result

    def test_provider_cost_estimation(self):
        """Test cost estimation."""
        provider = MockLLMProvider("test-mock")

        cost = provider.estimate_cost("Test prompt", max_tokens=100)
        assert isinstance(cost, float)
        assert cost > 0

    def test_provider_usage_stats(self):
        """Test usage statistics tracking."""
        provider = MockLLMProvider("test-mock")

        # Initial stats
        stats = provider.get_usage_stats()
        assert stats.total_requests == 0

        # Make a request
        provider.generate_completion("Test prompt")

        # Check updated stats
        stats = provider.get_usage_stats()
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.total_tokens_used > 0

    def test_provider_failure_simulation(self):
        """Test failure simulation."""
        provider = MockLLMProvider("test-mock", simulate_failures=True)

        # Make several requests to trigger simulated failure
        success_count = 0
        failure_count = 0

        for i in range(10):
            try:
                provider.generate_completion(f"Test prompt {i}")
                success_count += 1
            except LLMProviderError:
                failure_count += 1

        # Should have some failures when simulation is enabled
        assert failure_count > 0
        assert success_count > 0


class TestLLMServiceManager:
    """Test LLM service manager."""

    def test_service_manager_initialization(self):
        """Test service manager initialization."""
        config = LLMConfiguration()
        manager = LLMServiceManager(config)

        assert manager.config == config
        assert len(manager.providers) == 0
        assert len(manager.get_available_providers()) == 0

    def test_provider_management(self):
        """Test adding and removing providers."""
        config = LLMConfiguration()
        manager = LLMServiceManager(config)

        # Add provider
        provider = MockLLMProvider("test-provider")
        manager.add_provider(provider)

        assert "test-provider" in manager.providers
        assert "test-provider" in manager.get_available_providers()

        # Set as default
        manager.set_default_provider("test-provider")
        assert manager.default_provider == "test-provider"

        # Remove provider
        manager.remove_provider("test-provider")
        assert "test-provider" not in manager.providers
        assert len(manager.get_available_providers()) == 0

    def test_completion_generation(self):
        """Test completion generation through service manager."""
        config = LLMConfiguration()
        manager = LLMServiceManager(config)

        # Add provider
        provider = MockLLMProvider("test-provider")
        manager.add_provider(provider)
        manager.set_default_provider("test-provider")

        # Generate completion
        response = manager.generate_completion("Test prompt")
        assert isinstance(response, LLMResponse)
        assert response.provider == "test-provider"

    def test_structured_output_generation(self):
        """Test structured output generation."""
        config = LLMConfiguration()
        manager = LLMServiceManager(config)

        # Add provider
        provider = MockLLMProvider("test-provider")
        manager.add_provider(provider)
        manager.set_default_provider("test-provider")

        # Generate structured output
        schema = {"sentiment": "string", "confidence": "number"}
        result = manager.generate_structured_output("Test prompt", schema)
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "confidence" in result

    def test_fallback_mechanism(self):
        """Test fallback to secondary providers."""
        config = LLMConfiguration(fallback_enabled=True)
        manager = LLMServiceManager(config)

        # Add primary provider that will fail
        failing_provider = MockLLMProvider("failing-provider", simulate_failures=True)
        failing_provider.set_available(False)
        manager.add_provider(failing_provider)

        # Add backup provider
        backup_provider = MockLLMProvider("backup-provider")
        manager.add_provider(backup_provider)

        manager.set_default_provider("failing-provider")

        # Should fallback to backup provider
        response = manager.generate_completion("Test prompt")
        assert response.provider == "backup-provider"

    def test_budget_limits(self):
        """Test budget limit enforcement."""
        config = LLMConfiguration(cost_limit_per_hour=0.001)  # Very low limit
        manager = LLMServiceManager(config)

        # Add provider
        provider = MockLLMProvider("test-provider")
        manager.add_provider(provider)
        manager.set_default_provider("test-provider")

        # First request should work
        response1 = manager.generate_completion("Test prompt 1")
        assert isinstance(response1, LLMResponse)

        # Update hourly cost to exceed limit
        manager.total_hourly_cost = 0.002

        # Second request should fail due to budget
        with pytest.raises(LLMBudgetExceededError):
            manager.generate_completion("Test prompt 2")

    def test_service_disabled(self):
        """Test behavior when service is disabled."""
        config = LLMConfiguration(enabled=False)
        manager = LLMServiceManager(config)

        # Add provider
        provider = MockLLMProvider("test-provider")
        manager.add_provider(provider)
        manager.set_default_provider("test-provider")

        # Should raise error when disabled
        with pytest.raises(LLMServiceError):
            manager.generate_completion("Test prompt")

    def test_no_available_providers(self):
        """Test behavior when no providers are available."""
        config = LLMConfiguration()
        manager = LLMServiceManager(config)

        # No providers added
        with pytest.raises(LLMServiceError):
            manager.generate_completion("Test prompt")

    def test_usage_statistics(self):
        """Test usage statistics collection."""
        config = LLMConfiguration()
        manager = LLMServiceManager(config)

        # Add provider
        provider = MockLLMProvider("test-provider")
        manager.add_provider(provider)
        manager.set_default_provider("test-provider")

        # Make requests
        manager.generate_completion("Test prompt 1")
        manager.generate_completion("Test prompt 2")

        # Check stats
        stats = manager.get_usage_stats()
        assert "test-provider" in stats
        assert stats["test-provider"].total_requests == 2
        assert stats["test-provider"].successful_requests == 2

    def test_cost_estimation(self):
        """Test cost estimation."""
        config = LLMConfiguration()
        manager = LLMServiceManager(config)

        # Add provider
        provider = MockLLMProvider("test-provider")
        manager.add_provider(provider)
        manager.set_default_provider("test-provider")

        # Estimate cost
        cost = manager.estimate_request_cost("Test prompt")
        assert isinstance(cost, float)
        assert cost > 0
