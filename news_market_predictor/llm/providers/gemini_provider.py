"""
Google Gemini LLM provider implementation.

This provider integrates with Google's Gemini API, handling:
- API authentication and requests
- Rate limiting (15 requests/minute, 1500 requests/day for free tier)
- Token counting and cost estimation
- Error handling and retries
"""

import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..interfaces import LLMProvider
from ..models import LLMResponse, LLMUsageStats
from ..exceptions import (
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMInvalidResponseError,
)


@dataclass
class GeminiConfig:
    """Configuration for Google Gemini provider."""

    api_key: str
    model: str = "gemini-1.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    # Free tier limits
    max_requests_per_minute: int = 15
    max_requests_per_day: int = 1500

    # Cost estimation (free tier is $0, but we track for potential paid usage)
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider with free tier rate limiting and cost tracking.

    Features:
    - Free tier rate limiting (15 req/min, 1500 req/day)
    - Token counting and cost estimation
    - Automatic retries with exponential backoff
    - Structured output support via JSON mode
    - Error handling for API failures
    """

    def __init__(self, config: GeminiConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Usage tracking for rate limiting
        self.usage_stats = LLMUsageStats(provider="gemini")
        self.request_timestamps: List[float] = []
        self.daily_request_count = 0
        self.last_reset_date = datetime.now().date()

        # Validate configuration
        if not self.config.api_key:
            raise LLMProviderError("Gemini API key is required", provider="gemini")

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "gemini"

    def is_available(self) -> bool:
        """Check if the provider is currently available."""
        try:
            # Check if we're within rate limits
            if not self._can_make_request():
                return False

            # Simple health check - just verify API key format
            return bool(self.config.api_key and len(self.config.api_key) > 10)
        except Exception as e:
            self.logger.warning(f"Gemini availability check failed: {e}")
            return False

    def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a text completion from Gemini."""
        if not self._can_make_request():
            raise LLMRateLimitError(
                f"Gemini rate limit exceeded. Daily: {self.daily_request_count}/{self.config.max_requests_per_day}, "
                f"Recent requests: {len(self._get_recent_requests())}/{self.config.max_requests_per_minute}",
                provider="gemini",
            )

        model_name = model or self.config.model

        # Prepare request payload
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens or 2048,
                "temperature": temperature if temperature is not None else 0.1,
                "topP": kwargs.get("top_p", 0.8),
                "topK": kwargs.get("top_k", 40),
            },
        }

        start_time = time.time()

        try:
            response = self._make_api_request(model_name, payload)
            response_time = time.time() - start_time

            # Parse response
            content = self._extract_content(response)
            tokens_used = self._estimate_tokens(prompt, content)
            cost = self._calculate_cost(tokens_used, len(prompt.split()))

            # Create LLM response
            llm_response = LLMResponse(
                content=content,
                model_used=model_name,
                provider="gemini",
                tokens_used=tokens_used,
                cost_estimate=cost,
                confidence_score=0.8,  # Default confidence for Gemini
                reasoning_chain=[],
                metadata={
                    "response_time": response_time,
                    "api_response": response,
                    "generation_config": payload["generationConfig"],
                },
            )

            # Update usage stats
            self._record_successful_request(llm_response, response_time)

            return llm_response

        except LLMRateLimitError:
            self._record_failed_request()
            raise  # Re-raise rate limit errors as-is
        except requests.exceptions.Timeout:
            self._record_failed_request()
            raise LLMTimeoutError(
                f"Gemini request timed out after {self.config.timeout_seconds}s",
                provider="gemini",
                timeout_seconds=self.config.timeout_seconds,
            )
        except requests.exceptions.RequestException as e:
            self._record_failed_request()
            if "429" in str(e) or "quota" in str(e).lower():
                raise LLMRateLimitError(
                    f"Gemini rate limit exceeded: {e}", provider="gemini"
                )
            else:
                raise LLMProviderError(
                    f"Gemini API request failed: {e}", provider="gemini"
                )
        except Exception as e:
            self._record_failed_request()
            raise LLMProviderError(
                f"Unexpected error in Gemini provider: {e}", provider="gemini"
            )

    def generate_structured_output(
        self, prompt: str, schema: Dict[str, Any], model: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate structured output matching the provided schema."""
        # Enhance prompt with JSON schema instructions
        json_prompt = f"""
{prompt}

Please respond with valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

Respond only with the JSON object, no additional text or formatting.
"""

        # Generate completion
        response = self.generate_completion(
            json_prompt,
            model=model,
            temperature=kwargs.get(
                "temperature", 0.1
            ),  # Lower temperature for structured output
            **kwargs,
        )

        # Parse JSON response
        try:
            result = json.loads(response.content.strip())

            # Basic schema validation (check if required keys exist)
            if "properties" in schema:
                required_keys = schema.get("required", [])
                missing_keys = [key for key in required_keys if key not in result]
                if missing_keys:
                    raise LLMInvalidResponseError(
                        f"Missing required keys in response: {missing_keys}",
                        provider="gemini",
                        response_data=response.content,
                    )

            return result

        except json.JSONDecodeError as e:
            raise LLMInvalidResponseError(
                f"Invalid JSON response from Gemini: {e}",
                provider="gemini",
                response_data=response.content,
            )

    def estimate_cost(
        self, prompt: str, max_tokens: Optional[int] = None, model: Optional[str] = None
    ) -> float:
        """Estimate the cost of a request before making it."""
        # For free tier, cost is always 0
        if self.config.cost_per_1k_input_tokens == 0.0:
            return 0.0

        # Estimate tokens
        input_tokens = len(prompt.split()) * 1.3  # Rough estimation
        output_tokens = max_tokens or 1000

        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output_tokens

        return input_cost + output_cost

    def get_usage_stats(self) -> LLMUsageStats:
        """Get current usage statistics for this provider."""
        self._reset_daily_count_if_needed()

        # Update current stats
        recent_requests = self._get_recent_requests()
        self.usage_stats.requests_this_minute = len(recent_requests)
        self.usage_stats.requests_this_hour = len(
            [ts for ts in self.request_timestamps if time.time() - ts < 3600]
        )

        return self.usage_stats

    def reset_usage_stats(self) -> None:
        """Reset usage statistics (typically called hourly)."""
        self.usage_stats = LLMUsageStats(provider="gemini")
        self.request_timestamps.clear()
        self.daily_request_count = 0
        self.last_reset_date = datetime.now().date()

    def _can_make_request(self) -> bool:
        """Check if we can make a request within rate limits."""
        self._reset_daily_count_if_needed()

        # Check daily limit
        if self.daily_request_count >= self.config.max_requests_per_day:
            return False

        # Check per-minute limit
        recent_requests = self._get_recent_requests()
        if len(recent_requests) >= self.config.max_requests_per_minute:
            return False

        return True

    def _get_recent_requests(self) -> List[float]:
        """Get requests made in the last minute."""
        now = time.time()
        cutoff = now - 60  # 1 minute ago

        # Clean old timestamps and return recent ones
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
        return self.request_timestamps

    def _reset_daily_count_if_needed(self):
        """Reset daily request count if it's a new day."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_request_count = 0
            self.last_reset_date = today

    def _make_api_request(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to Gemini with retries."""
        url = f"{self.config.base_url}/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
        }
        params = {"key": self.config.api_key}

        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit hit
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise LLMRateLimitError(
                        f"Gemini rate limit exceeded, retry after {retry_after}s",
                        provider="gemini",
                        retry_after=retry_after,
                    )
                else:
                    error_msg = (
                        f"Gemini API error {response.status_code}: {response.text}"
                    )
                    raise LLMProviderError(error_msg, provider="gemini")

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (
                        2**attempt
                    )  # Exponential backoff
                    self.logger.warning(
                        f"Gemini request failed, retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                continue

        # All retries failed
        raise LLMProviderError(
            f"Gemini API request failed after {self.config.max_retries} retries: {last_error}",
            provider="gemini",
        )

    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract text content from Gemini API response."""
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                raise LLMInvalidResponseError(
                    "No candidates in Gemini response",
                    provider="gemini",
                    response_data=json.dumps(response),
                )

            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            if not parts:
                raise LLMInvalidResponseError(
                    "No parts in Gemini response content",
                    provider="gemini",
                    response_data=json.dumps(response),
                )

            # Combine all text parts
            text_parts = []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])

            if not text_parts:
                raise LLMInvalidResponseError(
                    "No text parts in Gemini response",
                    provider="gemini",
                    response_data=json.dumps(response),
                )

            return "\n".join(text_parts)

        except KeyError as e:
            raise LLMInvalidResponseError(
                f"Invalid Gemini response structure: missing {e}",
                provider="gemini",
                response_data=json.dumps(response),
            )

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count for prompt and response."""
        # Rough estimation: 1 token ≈ 0.75 words
        prompt_words = len(prompt.split())
        response_words = len(response.split())
        total_words = prompt_words + response_words
        return int(total_words * 1.33)  # Convert words to approximate tokens

    def _calculate_cost(self, tokens_used: int, input_word_count: int) -> float:
        """Calculate cost based on token usage."""
        # For free tier, cost is always 0
        if self.config.cost_per_1k_input_tokens == 0.0:
            return 0.0

        # Estimate input vs output tokens
        input_tokens = int(input_word_count * 1.33)
        output_tokens = tokens_used - input_tokens

        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output_tokens

        return max(0.0, input_cost + output_cost)

    def _record_successful_request(self, response: LLMResponse, response_time: float):
        """Record a successful request for usage tracking."""
        now = time.time()
        self.request_timestamps.append(now)
        self.daily_request_count += 1

        # Update usage stats
        self.usage_stats.total_requests += 1
        self.usage_stats.successful_requests += 1
        self.usage_stats.total_tokens_used += response.tokens_used
        self.usage_stats.total_cost += response.cost_estimate
        self.usage_stats.last_request_at = datetime.now()

        # Update average response time
        if self.usage_stats.total_requests > 1:
            self.usage_stats.average_response_time = (
                self.usage_stats.average_response_time
                * (self.usage_stats.total_requests - 1)
                + response_time
            ) / self.usage_stats.total_requests
        else:
            self.usage_stats.average_response_time = response_time

    def _record_failed_request(self):
        """Record a failed request for usage tracking."""
        now = time.time()
        self.request_timestamps.append(now)
        self.daily_request_count += 1

        # Update usage stats
        self.usage_stats.total_requests += 1
        self.usage_stats.failed_requests += 1
        self.usage_stats.last_request_at = datetime.now()
