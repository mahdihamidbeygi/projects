"""
LLM service manager with provider selection, fallback, and resource management.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque

from .interfaces import LLMService, LLMProvider
from .models import LLMResponse, LLMUsageStats, LLMConfiguration
from .exceptions import (
    LLMServiceError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMBudgetExceededError,
    LLMConfigurationError,
    LLMInvalidResponseError,
)


class CircuitBreaker:
    """Circuit breaker pattern implementation for provider reliability."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise LLMProviderError("Circuit breaker is OPEN - provider unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RateLimiter:
    """Rate limiter for controlling request frequency."""

    def __init__(
        self,
        max_per_minute: int = 60,
        max_per_hour: int = 1000,
        max_per_day: int = None,
    ):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self.minute_requests = deque()
        self.hour_requests = deque()
        self.day_requests = deque()

    def can_make_request(self) -> bool:
        """Check if a request can be made within rate limits."""
        now = time.time()
        self._cleanup_old_requests(now)

        # Check minute limit
        if len(self.minute_requests) >= self.max_per_minute:
            return False

        # Check hour limit
        if len(self.hour_requests) >= self.max_per_hour:
            return False

        # Check daily limit if configured
        if self.max_per_day and len(self.day_requests) >= self.max_per_day:
            return False

        return True

    def record_request(self):
        """Record a new request."""
        now = time.time()
        self.minute_requests.append(now)
        self.hour_requests.append(now)
        if self.max_per_day:
            self.day_requests.append(now)

    def get_time_until_next_request(self) -> float:
        """Get time in seconds until next request can be made."""
        now = time.time()
        self._cleanup_old_requests(now)

        # Check which limit is blocking
        if len(self.minute_requests) >= self.max_per_minute:
            # Wait until oldest minute request expires
            return max(0, 60 - (now - self.minute_requests[0]))
        elif len(self.hour_requests) >= self.max_per_hour:
            # Wait until oldest hour request expires
            return max(0, 3600 - (now - self.hour_requests[0]))
        elif self.max_per_day and len(self.day_requests) >= self.max_per_day:
            # Wait until oldest day request expires
            return max(0, 86400 - (now - self.day_requests[0]))

        return 0.0

    def get_current_usage(self) -> Dict[str, int]:
        """Get current usage counts."""
        now = time.time()
        self._cleanup_old_requests(now)

        return {
            "requests_this_minute": len(self.minute_requests),
            "requests_this_hour": len(self.hour_requests),
            "requests_this_day": len(self.day_requests) if self.max_per_day else 0,
            "max_per_minute": self.max_per_minute,
            "max_per_hour": self.max_per_hour,
            "max_per_day": self.max_per_day or 0,
        }

    def _cleanup_old_requests(self, now: float):
        """Remove old requests from tracking."""
        # Remove requests older than 1 minute
        while self.minute_requests and now - self.minute_requests[0] > 60:
            self.minute_requests.popleft()

        # Remove requests older than 1 hour
        while self.hour_requests and now - self.hour_requests[0] > 3600:
            self.hour_requests.popleft()

        # Remove requests older than 1 day
        if self.max_per_day:
            while self.day_requests and now - self.day_requests[0] > 86400:
                self.day_requests.popleft()


class LLMServiceManager(LLMService):
    """
    Provider-agnostic LLM service manager with fallback mechanisms.

    Features:
    - Multiple provider support with automatic fallback
    - Circuit breaker pattern for failed providers
    - Rate limiting and cost management
    - Usage tracking and budget controls
    - Load balancing across providers
    - Gemini as primary, Ollama as fallback configuration
    """

    def __init__(self, config: LLMConfiguration):
        self.config = config
        self.providers: Dict[str, LLMProvider] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.usage_stats: Dict[str, LLMUsageStats] = {}
        self.default_provider = config.default_provider
        self.logger = logging.getLogger(__name__)

        # Cost tracking
        self.hourly_cost_reset_time = datetime.now()
        self.total_hourly_cost = 0.0

        # Provider priority for load balancing (lower number = higher priority)
        # Configure Gemini as primary (priority 1), Ollama as fallback (priority 2)
        self.provider_priorities: Dict[str, int] = {
            "gemini": 1,  # Primary provider
            "ollama": 2,  # Fallback provider
        }

        # Provider selection strategy
        self.fallback_chain = ["gemini", "ollama"]  # Ordered fallback chain

    def add_provider(self, provider: LLMProvider) -> None:
        """Add an LLM provider to the service."""
        provider_name = provider.get_provider_name()

        if not provider_name:
            raise LLMConfigurationError("Provider must have a valid name")

        self.providers[provider_name] = provider
        self.circuit_breakers[provider_name] = CircuitBreaker()

        # Configure rate limiter based on provider type
        if provider_name == "gemini":
            # Gemini free tier: 15 requests/minute, 1500 requests/day
            self.rate_limiters[provider_name] = RateLimiter(
                max_per_minute=15,
                max_per_hour=min(1500, self.config.max_requests_per_hour),
                max_per_day=1500,
            )
        else:
            # Default rate limits for other providers (no daily limit)
            self.rate_limiters[provider_name] = RateLimiter(
                max_per_minute=self.config.max_requests_per_minute,
                max_per_hour=self.config.max_requests_per_hour,
            )

        self.usage_stats[provider_name] = LLMUsageStats(provider=provider_name)

        # Set priority based on provider type
        if provider_name not in self.provider_priorities:
            if provider_name == "gemini":
                self.provider_priorities[provider_name] = 1  # Primary
            elif provider_name == "ollama":
                self.provider_priorities[provider_name] = 2  # Fallback
            else:
                self.provider_priorities[provider_name] = 999  # Low priority

        self.logger.info(
            f"Added LLM provider: {provider_name} with priority {self.provider_priorities[provider_name]}"
        )

        # Update fallback chain if needed
        if provider_name not in self.fallback_chain:
            if provider_name == "gemini":
                self.fallback_chain.insert(0, provider_name)  # Insert at beginning
            elif provider_name == "ollama":
                if "ollama" not in self.fallback_chain:
                    self.fallback_chain.append(provider_name)  # Add at end
            else:
                self.fallback_chain.append(provider_name)  # Add other providers at end

    def remove_provider(self, provider_name: str) -> None:
        """Remove an LLM provider from the service."""
        if provider_name not in self.providers:
            raise LLMConfigurationError(f"Provider {provider_name} not found")

        del self.providers[provider_name]
        del self.circuit_breakers[provider_name]
        del self.rate_limiters[provider_name]
        del self.usage_stats[provider_name]
        del self.provider_priorities[provider_name]

        # Remove from fallback chain
        if provider_name in self.fallback_chain:
            self.fallback_chain.remove(provider_name)

        # Update default provider if removed
        if self.default_provider == provider_name:
            available_providers = list(self.providers.keys())
            self.default_provider = (
                available_providers[0] if available_providers else None
            )

        self.logger.info(f"Removed LLM provider: {provider_name}")

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [
            name
            for name, provider in self.providers.items()
            if provider.is_available() and self.circuit_breakers[name].state != "OPEN"
        ]

    def set_default_provider(self, provider_name: str) -> None:
        """Set the default provider to use."""
        if provider_name not in self.providers:
            raise LLMConfigurationError(f"Provider {provider_name} not found")

        self.default_provider = provider_name
        self.logger.info(f"Set default provider to: {provider_name}")

    def generate_completion(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion using the specified or default provider."""
        if not self.config.enabled:
            raise LLMServiceError("LLM service is disabled")

        # Check budget limits
        if not self.check_budget_limits():
            raise LLMBudgetExceededError(
                f"Budget limit exceeded: ${self.total_hourly_cost:.2f} >= ${self.config.cost_limit_per_hour:.2f}"
            )

        # Select provider
        selected_provider = self._select_provider(provider)

        # Try primary provider first, then fallback with retry logic
        providers_to_try = [selected_provider]
        if self.config.fallback_enabled:
            available_providers = self.get_available_providers()
            providers_to_try.extend(
                [p for p in available_providers if p != selected_provider]
            )

        last_error = None
        for provider_name in providers_to_try:
            try:
                return self._make_completion_request_with_retry(
                    provider_name, prompt, model, **kwargs
                )
            except LLMRateLimitError as e:
                last_error = e
                self.logger.warning(f"Provider {provider_name} rate limited: {e}")

                # If this is Gemini and we have Ollama available, try immediate fallback
                if provider_name == "gemini" and "ollama" in self.providers:
                    self.logger.info("Gemini quota exceeded, falling back to Ollama")
                    continue

                # For other rate limits, wait if it's the last provider
                if provider_name == providers_to_try[-1] and hasattr(e, "retry_after"):
                    self._handle_rate_limit_backoff(e.retry_after)

            except (LLMTimeoutError, LLMProviderError) as e:
                last_error = e
                self.logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        # All providers failed
        if last_error:
            raise last_error
        else:
            raise LLMServiceError("No available providers")

    def generate_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        provider: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate structured output using the specified or default provider."""
        if not self.config.enabled:
            raise LLMServiceError("LLM service is disabled")

        # Check budget limits
        if not self.check_budget_limits():
            raise LLMBudgetExceededError(
                f"Budget limit exceeded: ${self.total_hourly_cost:.2f} >= ${self.config.cost_limit_per_hour:.2f}"
            )

        # Select provider
        selected_provider = self._select_provider(provider)

        # Try primary provider first, then fallback with retry logic
        providers_to_try = [selected_provider]
        if self.config.fallback_enabled:
            available_providers = self.get_available_providers()
            providers_to_try.extend(
                [p for p in available_providers if p != selected_provider]
            )

        last_error = None
        for provider_name in providers_to_try:
            try:
                return self._make_structured_request_with_retry(
                    provider_name, prompt, schema, **kwargs
                )
            except LLMRateLimitError as e:
                last_error = e
                self.logger.warning(f"Provider {provider_name} rate limited: {e}")

                # If this is Gemini and we have Ollama available, try immediate fallback
                if provider_name == "gemini" and "ollama" in self.providers:
                    self.logger.info("Gemini quota exceeded, falling back to Ollama")
                    continue

                # For other rate limits, wait if it's the last provider
                if provider_name == providers_to_try[-1] and hasattr(e, "retry_after"):
                    self._handle_rate_limit_backoff(e.retry_after)

            except (LLMTimeoutError, LLMProviderError) as e:
                last_error = e
                self.logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        # All providers failed
        if last_error:
            raise last_error
        else:
            raise LLMServiceError("No available providers")

    def _make_structured_request_with_retry(
        self, provider_name: str, prompt: str, schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Make a structured output request with retry logic and exponential backoff."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                result = self._make_structured_request(
                    provider_name, prompt, schema, **kwargs
                )

                # Validate structured output
                if self._validate_structured_output(result, schema):
                    return result
                else:
                    raise LLMInvalidResponseError(
                        "Structured output validation failed",
                        provider=provider_name,
                        response_data=str(result),
                    )

            except LLMRateLimitError as e:
                # Don't retry rate limit errors, let the caller handle fallback
                raise e

            except (LLMTimeoutError, LLMProviderError, LLMInvalidResponseError) as e:
                last_error = e

                # Don't retry on the last attempt
                if attempt >= self.config.max_retries:
                    break

                # Calculate exponential backoff delay
                if self.config.exponential_backoff:
                    delay = self.config.retry_delay_seconds * (2**attempt)
                else:
                    delay = self.config.retry_delay_seconds

                # Cap the delay at 60 seconds
                delay = min(delay, 60.0)

                self.logger.warning(
                    f"Structured request to {provider_name} failed (attempt {attempt + 1}/{self.config.max_retries + 1}), "
                    f"retrying in {delay:.1f}s: {e}"
                )

                time.sleep(delay)

        # All retries exhausted
        raise last_error

    def _validate_structured_output(
        self, output: Dict[str, Any], schema: Dict[str, Any]
    ) -> bool:
        """Validate structured output against schema."""
        try:
            # Basic type check
            if not isinstance(output, dict):
                return False

            # Check required fields if specified in schema
            if "required" in schema:
                required_fields = schema["required"]
                for field in required_fields:
                    if field not in output:
                        self.logger.warning(f"Missing required field: {field}")
                        return False

            # Check if output is empty
            if not output:
                self.logger.warning("Empty structured output")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating structured output: {e}")
            return False

    def is_available(self, provider: Optional[str] = None) -> bool:
        """Check if the specified provider (or any provider) is available."""
        if provider:
            return (
                provider in self.providers
                and self.providers[provider].is_available()
                and self.circuit_breakers[provider].state != "OPEN"
            )
        else:
            return len(self.get_available_providers()) > 0

    def get_usage_stats(
        self, provider: Optional[str] = None
    ) -> Dict[str, LLMUsageStats]:
        """Get usage statistics for specified provider or all providers."""
        if provider:
            if provider not in self.usage_stats:
                raise LLMConfigurationError(f"Provider {provider} not found")
            return {provider: self.usage_stats[provider]}
        else:
            return self.usage_stats.copy()

    def get_rate_limit_status(
        self, provider: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get current rate limit status for providers."""
        if provider:
            if provider not in self.rate_limiters:
                raise LLMConfigurationError(f"Provider {provider} not found")
            return {provider: self.rate_limiters[provider].get_current_usage()}
        else:
            return {
                name: limiter.get_current_usage()
                for name, limiter in self.rate_limiters.items()
            }

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary across all providers."""
        total_cost = sum(stats.total_cost for stats in self.usage_stats.values())
        hourly_cost = self.total_hourly_cost

        provider_costs = {
            name: {
                "total_cost": stats.total_cost,
                "cost_this_hour": stats.cost_this_hour,
                "total_requests": stats.total_requests,
                "total_tokens": stats.total_tokens_used,
            }
            for name, stats in self.usage_stats.items()
        }

        return {
            "total_cost_all_time": total_cost,
            "cost_this_hour": hourly_cost,
            "cost_limit_per_hour": self.config.cost_limit_per_hour,
            "budget_remaining": max(0, self.config.cost_limit_per_hour - hourly_cost),
            "budget_utilization_percent": (
                hourly_cost / self.config.cost_limit_per_hour
            )
            * 100,
            "provider_breakdown": provider_costs,
            "next_reset_time": (
                self.hourly_cost_reset_time + timedelta(hours=1)
            ).isoformat(),
        }

    def check_budget_limits(self) -> bool:
        """Check if current usage is within budget limits."""
        self._reset_hourly_costs_if_needed()
        return self.total_hourly_cost < self.config.cost_limit_per_hour

    def get_provider_priorities(self) -> Dict[str, int]:
        """Get current provider priorities."""
        return self.provider_priorities.copy()

    def get_fallback_chain(self) -> List[str]:
        """Get current fallback chain."""
        return self.fallback_chain.copy()

    def set_provider_priority(self, provider_name: str, priority: int) -> None:
        """Set priority for a specific provider."""
        if provider_name not in self.providers:
            raise LLMConfigurationError(f"Provider {provider_name} not found")

        self.provider_priorities[provider_name] = priority
        self.logger.info(f"Set priority for {provider_name} to {priority}")

    def estimate_request_cost(
        self, prompt: str, provider: Optional[str] = None, **kwargs
    ) -> float:
        """Estimate the cost of a request before making it."""
        selected_provider = self._select_provider(provider)
        if selected_provider not in self.providers:
            raise LLMConfigurationError(f"Provider {selected_provider} not found")

        return self.providers[selected_provider].estimate_cost(
            prompt,
            kwargs.get("max_tokens", self.config.max_tokens),
            kwargs.get("model", self.config.model_name),
        )

    def _select_provider(self, preferred_provider: Optional[str] = None) -> str:
        """Select the best available provider based on priority and availability."""
        if preferred_provider and preferred_provider in self.providers:
            # Check if preferred provider is available and not circuit broken
            if (
                self.providers[preferred_provider].is_available()
                and self.circuit_breakers[preferred_provider].state != "OPEN"
            ):
                return preferred_provider

        # Use fallback chain for provider selection
        for provider_name in self.fallback_chain:
            if (
                provider_name in self.providers
                and self.providers[provider_name].is_available()
                and self.circuit_breakers[provider_name].state != "OPEN"
            ):
                return provider_name

        # If no providers in fallback chain are available, try any available provider
        available_providers = self.get_available_providers()
        if not available_providers:
            raise LLMServiceError("No available providers")

        # Sort by priority (lower number = higher priority)
        available_providers.sort(key=lambda p: self.provider_priorities.get(p, 999))
        return available_providers[0]

    def _make_completion_request_with_retry(
        self, provider_name: str, prompt: str, model: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """Make a completion request with retry logic and exponential backoff."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                return self._make_completion_request(
                    provider_name, prompt, model, **kwargs
                )

            except LLMRateLimitError as e:
                # Don't retry rate limit errors, let the caller handle fallback
                raise e

            except (LLMTimeoutError, LLMProviderError) as e:
                last_error = e

                # Don't retry on the last attempt
                if attempt >= self.config.max_retries:
                    break

                # Calculate exponential backoff delay
                if self.config.exponential_backoff:
                    delay = self.config.retry_delay_seconds * (2**attempt)
                else:
                    delay = self.config.retry_delay_seconds

                # Cap the delay at 60 seconds
                delay = min(delay, 60.0)

                self.logger.warning(
                    f"Request to {provider_name} failed (attempt {attempt + 1}/{self.config.max_retries + 1}), "
                    f"retrying in {delay:.1f}s: {e}"
                )

                time.sleep(delay)

        # All retries exhausted
        raise last_error

    def _handle_rate_limit_backoff(self, retry_after: int):
        """Handle rate limit with exponential backoff."""
        # For rate limits, use the server-suggested retry time or a minimum delay
        delay = max(retry_after, 1)

        # Cap the delay at a reasonable maximum
        delay = min(delay, 300)  # Max 5 minutes

        self.logger.info(f"Rate limited, waiting {delay} seconds before retry")
        time.sleep(delay)

    def _validate_response(self, response: LLMResponse) -> bool:
        """Validate LLM response for completeness and quality."""
        try:
            # Basic validation
            if not response.content or not response.content.strip():
                self.logger.warning("Empty response content")
                return False

            # Check if response seems truncated
            if len(response.content) < 10:
                self.logger.warning("Response seems too short")
                return False

            # Check confidence score if available
            if response.confidence_score < 0.3:
                self.logger.warning(
                    f"Low confidence response: {response.confidence_score}"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating response: {e}")
            return False

    def _make_completion_request(
        self, provider_name: str, prompt: str, model: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """Make a completion request to a specific provider."""
        provider = self.providers[provider_name]
        circuit_breaker = self.circuit_breakers[provider_name]
        rate_limiter = self.rate_limiters[provider_name]

        # Check rate limits with detailed error information
        if not rate_limiter.can_make_request():
            usage = rate_limiter.get_current_usage()
            wait_time = rate_limiter.get_time_until_next_request()

            error_msg = (
                f"Rate limit exceeded for provider {provider_name}. "
                f"Current usage: {usage['requests_this_minute']}/{usage['max_per_minute']} per minute, "
                f"{usage['requests_this_hour']}/{usage['max_per_hour']} per hour"
            )

            if usage["max_per_day"] > 0:
                error_msg += (
                    f", {usage['requests_this_day']}/{usage['max_per_day']} per day"
                )

            if wait_time > 0:
                error_msg += f". Next request available in {wait_time:.1f} seconds"

            raise LLMRateLimitError(
                error_msg, provider=provider_name, retry_after=int(wait_time)
            )

        # Make request with circuit breaker protection
        start_time = time.time()
        try:
            response = circuit_breaker.call(
                provider.generate_completion,
                prompt,
                model or self.config.model_name,
                kwargs.get("max_tokens", self.config.max_tokens),
                kwargs.get("temperature", self.config.temperature),
                **kwargs,
            )

            # Validate response quality
            if not self._validate_response(response):
                raise LLMInvalidResponseError(
                    "Response validation failed",
                    provider=provider_name,
                    response_data=response.content,
                )

            # Record successful request
            rate_limiter.record_request()
            self._update_usage_stats(
                provider_name, response, time.time() - start_time, success=True
            )
            self._update_hourly_cost(response.cost_estimate)

            return response

        except Exception as e:
            # Record failed request
            self._update_usage_stats(
                provider_name, None, time.time() - start_time, success=False
            )
            raise e

    def _make_structured_request(
        self, provider_name: str, prompt: str, schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Make a structured output request to a specific provider."""
        provider = self.providers[provider_name]
        circuit_breaker = self.circuit_breakers[provider_name]
        rate_limiter = self.rate_limiters[provider_name]

        # Check rate limits with detailed error information
        if not rate_limiter.can_make_request():
            usage = rate_limiter.get_current_usage()
            wait_time = rate_limiter.get_time_until_next_request()

            error_msg = (
                f"Rate limit exceeded for provider {provider_name}. "
                f"Current usage: {usage['requests_this_minute']}/{usage['max_per_minute']} per minute, "
                f"{usage['requests_this_hour']}/{usage['max_per_hour']} per hour"
            )

            if usage["max_per_day"] > 0:
                error_msg += (
                    f", {usage['requests_this_day']}/{usage['max_per_day']} per day"
                )

            if wait_time > 0:
                error_msg += f". Next request available in {wait_time:.1f} seconds"

            raise LLMRateLimitError(
                error_msg, provider=provider_name, retry_after=int(wait_time)
            )

        # Make request with circuit breaker protection
        start_time = time.time()
        try:
            result = circuit_breaker.call(
                provider.generate_structured_output,
                prompt,
                schema,
                kwargs.get("model", self.config.model_name),
                **kwargs,
            )

            # Record successful request
            rate_limiter.record_request()
            # Note: structured output doesn't return LLMResponse, so we can't track detailed stats
            self._update_usage_stats(
                provider_name, None, time.time() - start_time, success=True
            )

            return result

        except Exception as e:
            # Record failed request
            self._update_usage_stats(
                provider_name, None, time.time() - start_time, success=False
            )
            raise e

    def _update_usage_stats(
        self,
        provider_name: str,
        response: Optional[LLMResponse],
        response_time: float,
        success: bool,
    ):
        """Update usage statistics for a provider."""
        stats = self.usage_stats[provider_name]

        stats.total_requests += 1
        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1

        if response:
            stats.total_tokens_used += response.tokens_used
            stats.total_cost += response.cost_estimate

        # Update average response time
        if stats.total_requests > 1:
            stats.average_response_time = (
                stats.average_response_time * (stats.total_requests - 1) + response_time
            ) / stats.total_requests
        else:
            stats.average_response_time = response_time

        stats.last_request_at = datetime.now()

    def _update_hourly_cost(self, cost: float):
        """Update hourly cost tracking."""
        self._reset_hourly_costs_if_needed()
        self.total_hourly_cost += cost

    def _reset_hourly_costs_if_needed(self):
        """Reset hourly cost tracking if an hour has passed."""
        now = datetime.now()
        if now - self.hourly_cost_reset_time >= timedelta(hours=1):
            self.total_hourly_cost = 0.0
            self.hourly_cost_reset_time = now

            # Also reset provider usage stats
            for stats in self.usage_stats.values():
                stats.requests_this_minute = 0
                stats.requests_this_hour = 0
                stats.cost_this_hour = 0.0
