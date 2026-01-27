"""
Ollama LLM provider implementation for local model inference.

This provider integrates with Ollama local API, providing:
- Local model support (Llama 3.2, Mistral, etc.)
- Unlimited usage (no rate limits or costs)
- Consistent interface with other providers
- Fallback option when cloud providers are unavailable
"""

import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..interfaces import LLMProvider
from ..models import LLMResponse, LLMUsageStats
from ..exceptions import (
    LLMProviderError,
    LLMTimeoutError,
    LLMInvalidResponseError,
)


@dataclass
class OllamaConfig:
    """Configuration for Ollama provider."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout_seconds: int = 60  # Longer timeout for local inference
    max_retries: int = 2
    retry_delay: float = 2.0

    # Available models to check
    available_models: List[str] = None

    def __post_init__(self):
        if self.available_models is None:
            self.available_models = [
                "llama3.2",
                "llama3.2:1b",
                "llama3.2:3b",
                "mistral",
                "mistral:7b",
                "codellama",
                "phi3",
                "gemma2",
            ]


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for local model inference.

    Features:
    - Local model support with no rate limits
    - Unlimited usage (no costs)
    - Multiple model support (Llama, Mistral, etc.)
    - Automatic model availability checking
    - Consistent interface with cloud providers
    """

    def __init__(self, config: OllamaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Usage tracking (no limits, but track for statistics)
        self.usage_stats = LLMUsageStats(provider="ollama")

        # Check if Ollama is running
        self._check_ollama_connection()

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "ollama"

    def is_available(self) -> bool:
        """Check if the provider is currently available."""
        try:
            # Check if Ollama service is running
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ollama availability check failed: {e}")
            return False

    def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a text completion from Ollama."""
        model_name = model or self.config.model

        # Ensure model is available
        if not self._is_model_available(model_name):
            # Try to pull the model
            self.logger.info(f"Model {model_name} not found, attempting to pull...")
            if not self._pull_model(model_name):
                raise LLMProviderError(
                    f"Model {model_name} is not available and could not be pulled",
                    provider="ollama",
                )

        # Prepare request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,  # We want complete response
            "options": {
                "temperature": temperature if temperature is not None else 0.1,
                "num_predict": max_tokens or 2048,
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
            },
        }

        start_time = time.time()

        try:
            response = self._make_api_request("/api/generate", payload)
            response_time = time.time() - start_time

            # Parse response
            content = response.get("response", "")
            if not content:
                raise LLMInvalidResponseError(
                    "Empty response from Ollama",
                    provider="ollama",
                    response_data=json.dumps(response),
                )

            # Extract token information
            tokens_used = self._calculate_tokens(response)

            # Create LLM response
            llm_response = LLMResponse(
                content=content,
                model_used=model_name,
                provider="ollama",
                tokens_used=tokens_used,
                cost_estimate=0.0,  # Local inference is free
                confidence_score=0.7,  # Default confidence for local models
                reasoning_chain=[],
                metadata={
                    "response_time": response_time,
                    "eval_count": response.get("eval_count", 0),
                    "eval_duration": response.get("eval_duration", 0),
                    "prompt_eval_count": response.get("prompt_eval_count", 0),
                    "prompt_eval_duration": response.get("prompt_eval_duration", 0),
                    "total_duration": response.get("total_duration", 0),
                    "load_duration": response.get("load_duration", 0),
                },
            )

            # Update usage stats
            self._record_successful_request(llm_response, response_time)

            return llm_response

        except requests.exceptions.Timeout:
            self._record_failed_request()
            raise LLMTimeoutError(
                f"Ollama request timed out after {self.config.timeout_seconds}s",
                provider="ollama",
                timeout_seconds=self.config.timeout_seconds,
            )
        except requests.exceptions.RequestException as e:
            self._record_failed_request()
            raise LLMProviderError(f"Ollama API request failed: {e}", provider="ollama")
        except Exception as e:
            self._record_failed_request()
            raise LLMProviderError(
                f"Unexpected error in Ollama provider: {e}", provider="ollama"
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

Important: Respond ONLY with the JSON object. Do not include any additional text, explanations, or formatting outside the JSON.
"""

        # Generate completion with lower temperature for more consistent JSON
        response = self.generate_completion(
            json_prompt,
            model=model,
            temperature=kwargs.get("temperature", 0.1),
            **kwargs,
        )

        # Parse JSON response
        try:
            # Clean the response content
            content = response.content.strip()

            # Try to extract JSON if there's extra text
            if not content.startswith("{") and not content.startswith("["):
                # Look for JSON in the response
                start_idx = content.find("{")
                if start_idx == -1:
                    start_idx = content.find("[")
                if start_idx != -1:
                    content = content[start_idx:]

            # Find the end of JSON
            if content.startswith("{"):
                brace_count = 0
                end_idx = 0
                for i, char in enumerate(content):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                if end_idx > 0:
                    content = content[:end_idx]

            result = json.loads(content)

            # Basic schema validation (check if required keys exist)
            if "properties" in schema:
                required_keys = schema.get("required", [])
                missing_keys = [key for key in required_keys if key not in result]
                if missing_keys:
                    raise LLMInvalidResponseError(
                        f"Missing required keys in response: {missing_keys}",
                        provider="ollama",
                        response_data=response.content,
                    )

            return result

        except json.JSONDecodeError as e:
            raise LLMInvalidResponseError(
                f"Invalid JSON response from Ollama: {e}",
                provider="ollama",
                response_data=response.content,
            )

    def estimate_cost(
        self, prompt: str, max_tokens: Optional[int] = None, model: Optional[str] = None
    ) -> float:
        """Estimate the cost of a request before making it."""
        # Local inference is always free
        return 0.0

    def get_usage_stats(self) -> LLMUsageStats:
        """Get current usage statistics for this provider."""
        return self.usage_stats

    def reset_usage_stats(self) -> None:
        """Reset usage statistics (typically called hourly)."""
        self.usage_stats = LLMUsageStats(provider="ollama")

    def list_available_models(self) -> List[str]:
        """List models available in Ollama."""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return [model.get("name", "") for model in models if model.get("name")]
            else:
                self.logger.warning(
                    f"Failed to list Ollama models: {response.status_code}"
                )
                return []
        except Exception as e:
            self.logger.warning(f"Error listing Ollama models: {e}")
            return []

    def _check_ollama_connection(self):
        """Check if Ollama service is running and accessible."""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self.logger.warning(
                    f"Ollama service not accessible at {self.config.base_url}. "
                    "Make sure Ollama is installed and running."
                )
        except Exception as e:
            self.logger.warning(
                f"Cannot connect to Ollama at {self.config.base_url}: {e}. "
                "Make sure Ollama is installed and running."
            )

    def _is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available in Ollama."""
        available_models = self.list_available_models()

        # Check exact match first
        if model_name in available_models:
            return True

        # Check if model name matches any available model (with tags)
        for available_model in available_models:
            if available_model.startswith(model_name + ":") or model_name.startswith(
                available_model + ":"
            ):
                return True

        return False

    def _pull_model(self, model_name: str) -> bool:
        """Attempt to pull a model from Ollama registry."""
        try:
            self.logger.info(f"Pulling Ollama model: {model_name}")

            payload = {"name": model_name}
            response = requests.post(
                f"{self.config.base_url}/api/pull",
                json=payload,
                timeout=300,  # 5 minutes for model download
            )

            if response.status_code == 200:
                self.logger.info(f"Successfully pulled model: {model_name}")
                return True
            else:
                self.logger.error(
                    f"Failed to pull model {model_name}: {response.status_code}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return False

    def _make_api_request(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make API request to Ollama with retries."""
        url = f"{self.config.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
        }

        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = (
                        f"Ollama API error {response.status_code}: {response.text}"
                    )
                    raise LLMProviderError(error_msg, provider="ollama")

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (attempt + 1)
                    self.logger.warning(
                        f"Ollama request failed, retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                continue

        # All retries failed
        raise LLMProviderError(
            f"Ollama API request failed after {self.config.max_retries} retries: {last_error}",
            provider="ollama",
        )

    def _calculate_tokens(self, response: Dict[str, Any]) -> int:
        """Calculate token count from Ollama response metadata."""
        # Ollama provides token counts in the response
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)

        # If not available, estimate from text
        if prompt_tokens == 0 and completion_tokens == 0:
            response_text = response.get("response", "")
            return len(response_text.split()) * 1.3  # Rough estimation

        return prompt_tokens + completion_tokens

    def _record_successful_request(self, response: LLMResponse, response_time: float):
        """Record a successful request for usage tracking."""
        # Update usage stats
        self.usage_stats.total_requests += 1
        self.usage_stats.successful_requests += 1
        self.usage_stats.total_tokens_used += response.tokens_used
        self.usage_stats.total_cost += response.cost_estimate  # Always 0 for Ollama
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
        # Update usage stats
        self.usage_stats.total_requests += 1
        self.usage_stats.failed_requests += 1
        self.usage_stats.last_request_at = datetime.now()
