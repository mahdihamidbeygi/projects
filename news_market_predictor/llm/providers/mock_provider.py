"""
Mock LLM provider for testing and development.
"""

import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

from ..interfaces import LLMProvider
from ..models import LLMResponse, LLMUsageStats
from ..exceptions import LLMProviderError, LLMTimeoutError


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing and development purposes."""

    def __init__(self, provider_name: str = "mock", simulate_failures: bool = False):
        self.provider_name = provider_name
        self.simulate_failures = simulate_failures
        self.usage_stats = LLMUsageStats(provider=provider_name)
        self.available = True
        self.failure_count = 0

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return self.provider_name

    def is_available(self) -> bool:
        """Check if the provider is currently available."""
        return self.available

    def set_available(self, available: bool):
        """Set provider availability (for testing)."""
        self.available = available

    def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a mock text completion."""
        if not self.available:
            raise LLMProviderError(f"Provider {self.provider_name} is not available")

        # Simulate occasional failures if enabled
        if self.simulate_failures and self.failure_count % 5 == 4:
            self.failure_count += 1
            raise LLMTimeoutError(f"Simulated timeout from {self.provider_name}")

        self.failure_count += 1

        # Simulate processing time
        time.sleep(0.1)

        # Generate mock response
        mock_content = f"Mock response to: {prompt[:50]}..."
        tokens_used = len(prompt.split()) + len(mock_content.split())
        cost_estimate = tokens_used * 0.0001  # Mock cost calculation

        response = LLMResponse(
            content=mock_content,
            model_used=model or "mock-model",
            provider=self.provider_name,
            tokens_used=tokens_used,
            cost_estimate=cost_estimate,
            confidence_score=0.8,
            reasoning_chain=[
                "Analyzed input prompt",
                "Generated appropriate response",
                "Validated output format",
            ],
            metadata={
                "temperature": temperature or 0.1,
                "max_tokens": max_tokens or 2000,
                "mock_provider": True,
            },
        )

        # Update usage stats
        self.usage_stats.total_requests += 1
        self.usage_stats.successful_requests += 1
        self.usage_stats.total_tokens_used += tokens_used
        self.usage_stats.total_cost += cost_estimate
        self.usage_stats.last_request_at = datetime.now()

        return response

    def generate_structured_output(
        self, prompt: str, schema: Dict[str, Any], model: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate mock structured output matching the provided schema."""
        if not self.available:
            raise LLMProviderError(f"Provider {self.provider_name} is not available")

        # Simulate occasional failures if enabled
        if self.simulate_failures and self.failure_count % 7 == 6:
            self.failure_count += 1
            raise LLMProviderError(f"Simulated error from {self.provider_name}")

        self.failure_count += 1

        # Simulate processing time
        time.sleep(0.1)

        # Generate mock structured output based on schema
        mock_output = self._generate_mock_from_schema(schema)

        # Update usage stats
        self.usage_stats.total_requests += 1
        self.usage_stats.successful_requests += 1
        self.usage_stats.last_request_at = datetime.now()

        return mock_output

    def estimate_cost(
        self, prompt: str, max_tokens: Optional[int] = None, model: Optional[str] = None
    ) -> float:
        """Estimate the cost of a request."""
        estimated_tokens = len(prompt.split()) + (max_tokens or 100)
        return estimated_tokens * 0.0001  # Mock cost per token

    def get_usage_stats(self) -> LLMUsageStats:
        """Get current usage statistics for this provider."""
        return self.usage_stats

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = LLMUsageStats(provider=self.provider_name)

    def _generate_mock_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock data matching the provided schema."""
        result = {}

        for key, value_type in schema.items():
            if value_type == "string":
                result[key] = f"mock_{key}_value"
            elif value_type == "number":
                result[key] = 0.5
            elif value_type == "integer":
                result[key] = 42
            elif value_type == "boolean":
                result[key] = True
            elif value_type == "array":
                result[key] = ["mock_item_1", "mock_item_2"]
            elif isinstance(value_type, dict):
                result[key] = self._generate_mock_from_schema(value_type)
            else:
                result[key] = f"mock_{key}"

        return result
