"""
Abstract interfaces for LLM service layer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .models import LLMResponse, LLMUsageStats


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is currently available."""
        pass

    @abstractmethod
    def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a text completion from the LLM."""
        pass

    @abstractmethod
    def generate_structured_output(
        self, prompt: str, schema: Dict[str, Any], model: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate structured output matching the provided schema."""
        pass

    @abstractmethod
    def estimate_cost(
        self, prompt: str, max_tokens: Optional[int] = None, model: Optional[str] = None
    ) -> float:
        """Estimate the cost of a request before making it."""
        pass

    @abstractmethod
    def get_usage_stats(self) -> LLMUsageStats:
        """Get current usage statistics for this provider."""
        pass

    @abstractmethod
    def reset_usage_stats(self) -> None:
        """Reset usage statistics (typically called hourly)."""
        pass


class LLMService(ABC):
    """Abstract base class for LLM service management."""

    @abstractmethod
    def add_provider(self, provider: LLMProvider) -> None:
        """Add an LLM provider to the service."""
        pass

    @abstractmethod
    def remove_provider(self, provider_name: str) -> None:
        """Remove an LLM provider from the service."""
        pass

    @abstractmethod
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        pass

    @abstractmethod
    def set_default_provider(self, provider_name: str) -> None:
        """Set the default provider to use."""
        pass

    @abstractmethod
    def generate_completion(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a completion using the specified or default provider."""
        pass

    @abstractmethod
    def generate_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        provider: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured output using the specified or default provider."""
        pass

    @abstractmethod
    def is_available(self, provider: Optional[str] = None) -> bool:
        """Check if the specified provider (or any provider) is available."""
        pass

    @abstractmethod
    def get_usage_stats(
        self, provider: Optional[str] = None
    ) -> Dict[str, LLMUsageStats]:
        """Get usage statistics for specified provider or all providers."""
        pass

    @abstractmethod
    def check_budget_limits(self) -> bool:
        """Check if current usage is within budget limits."""
        pass

    @abstractmethod
    def estimate_request_cost(
        self, prompt: str, provider: Optional[str] = None, **kwargs
    ) -> float:
        """Estimate the cost of a request before making it."""
        pass
