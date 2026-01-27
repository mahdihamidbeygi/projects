"""
Data models for LLM service layer.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from ..models import ValidationError


@dataclass
class LLMResponse:
    """Response from an LLM service call."""

    content: str
    model_used: str
    provider: str
    tokens_used: int
    cost_estimate: float
    confidence_score: float
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> bool:
        """Validate LLMResponse data."""
        if not isinstance(self.content, str):
            raise ValidationError("Content must be a string")

        if not isinstance(self.model_used, str) or not self.model_used.strip():
            raise ValidationError("Model used must be a non-empty string")

        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValidationError("Provider must be a non-empty string")

        if not isinstance(self.tokens_used, int) or self.tokens_used < 0:
            raise ValidationError("Tokens used must be a non-negative integer")

        if not isinstance(self.cost_estimate, (int, float)) or self.cost_estimate < 0:
            raise ValidationError("Cost estimate must be a non-negative number")

        if not isinstance(self.confidence_score, (int, float)):
            raise ValidationError("Confidence score must be a number")

        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValidationError("Confidence score must be between 0.0 and 1.0")

        if not isinstance(self.reasoning_chain, list):
            raise ValidationError("Reasoning chain must be a list")

        if not all(isinstance(step, str) for step in self.reasoning_chain):
            raise ValidationError("All reasoning chain steps must be strings")

        if not isinstance(self.metadata, dict):
            raise ValidationError("Metadata must be a dictionary")

        if not isinstance(self.created_at, datetime):
            raise ValidationError("Created at must be a datetime object")

        return True

    def to_json(self) -> str:
        """Serialize LLMResponse to JSON format."""
        self.validate()
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "LLMResponse":
        """Deserialize LLMResponse from JSON format."""
        try:
            data = json.loads(json_str)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            response = cls(**data)
            response.validate()
            return response
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid JSON format for LLMResponse: {e}") from e


@dataclass
class LLMConfiguration:
    """Configuration for LLM services."""

    enabled: bool = True
    default_provider: str = "openai"
    model_name: str = "gpt-4"
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout_seconds: int = 30
    cost_limit_per_hour: float = 10.0
    fallback_enabled: bool = True

    # Provider-specific settings
    provider_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True

    def validate(self) -> bool:
        """Validate LLMConfiguration data."""
        if not isinstance(self.enabled, bool):
            raise ValidationError("Enabled must be a boolean")

        if (
            not isinstance(self.default_provider, str)
            or not self.default_provider.strip()
        ):
            raise ValidationError("Default provider must be a non-empty string")

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValidationError("Model name must be a non-empty string")

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValidationError("Max tokens must be a positive integer")

        if (
            not isinstance(self.temperature, (int, float))
            or not 0.0 <= self.temperature <= 2.0
        ):
            raise ValidationError("Temperature must be a number between 0.0 and 2.0")

        if not isinstance(self.timeout_seconds, int) or self.timeout_seconds <= 0:
            raise ValidationError("Timeout seconds must be a positive integer")

        if (
            not isinstance(self.cost_limit_per_hour, (int, float))
            or self.cost_limit_per_hour < 0
        ):
            raise ValidationError("Cost limit per hour must be a non-negative number")

        if not isinstance(self.fallback_enabled, bool):
            raise ValidationError("Fallback enabled must be a boolean")

        if not isinstance(self.provider_configs, dict):
            raise ValidationError("Provider configs must be a dictionary")

        if (
            not isinstance(self.max_requests_per_minute, int)
            or self.max_requests_per_minute <= 0
        ):
            raise ValidationError("Max requests per minute must be a positive integer")

        if (
            not isinstance(self.max_requests_per_hour, int)
            or self.max_requests_per_hour <= 0
        ):
            raise ValidationError("Max requests per hour must be a positive integer")

        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValidationError("Max retries must be a non-negative integer")

        if (
            not isinstance(self.retry_delay_seconds, (int, float))
            or self.retry_delay_seconds < 0
        ):
            raise ValidationError("Retry delay seconds must be a non-negative number")

        if not isinstance(self.exponential_backoff, bool):
            raise ValidationError("Exponential backoff must be a boolean")

        return True

    def to_json(self) -> str:
        """Serialize LLMConfiguration to JSON format."""
        self.validate()
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "LLMConfiguration":
        """Deserialize LLMConfiguration from JSON format."""
        try:
            data = json.loads(json_str)
            config = cls(**data)
            config.validate()
            return config
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid JSON format for LLMConfiguration: {e}"
            ) from e


@dataclass
class LLMUsageStats:
    """Usage statistics for LLM services."""

    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_request_at: Optional[datetime] = None
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    cost_this_hour: float = 0.0

    def validate(self) -> bool:
        """Validate LLMUsageStats data."""
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValidationError("Provider must be a non-empty string")

        if not isinstance(self.total_requests, int) or self.total_requests < 0:
            raise ValidationError("Total requests must be a non-negative integer")

        if (
            not isinstance(self.successful_requests, int)
            or self.successful_requests < 0
        ):
            raise ValidationError("Successful requests must be a non-negative integer")

        if not isinstance(self.failed_requests, int) or self.failed_requests < 0:
            raise ValidationError("Failed requests must be a non-negative integer")

        if self.successful_requests + self.failed_requests != self.total_requests:
            raise ValidationError(
                "Successful + failed requests must equal total requests"
            )

        if not isinstance(self.total_tokens_used, int) or self.total_tokens_used < 0:
            raise ValidationError("Total tokens used must be a non-negative integer")

        if not isinstance(self.total_cost, (int, float)) or self.total_cost < 0:
            raise ValidationError("Total cost must be a non-negative number")

        if (
            not isinstance(self.average_response_time, (int, float))
            or self.average_response_time < 0
        ):
            raise ValidationError("Average response time must be a non-negative number")

        if self.last_request_at is not None and not isinstance(
            self.last_request_at, datetime
        ):
            raise ValidationError("Last request at must be a datetime object or None")

        if (
            not isinstance(self.requests_this_minute, int)
            or self.requests_this_minute < 0
        ):
            raise ValidationError("Requests this minute must be a non-negative integer")

        if not isinstance(self.requests_this_hour, int) or self.requests_this_hour < 0:
            raise ValidationError("Requests this hour must be a non-negative integer")

        if not isinstance(self.cost_this_hour, (int, float)) or self.cost_this_hour < 0:
            raise ValidationError("Cost this hour must be a non-negative number")

        return True

    def to_json(self) -> str:
        """Serialize LLMUsageStats to JSON format."""
        self.validate()
        data = asdict(self)
        if self.last_request_at:
            data["last_request_at"] = self.last_request_at.isoformat()
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "LLMUsageStats":
        """Deserialize LLMUsageStats from JSON format."""
        try:
            data = json.loads(json_str)
            if data.get("last_request_at"):
                data["last_request_at"] = datetime.fromisoformat(
                    data["last_request_at"]
                )
            stats = cls(**data)
            stats.validate()
            return stats
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid JSON format for LLMUsageStats: {e}") from e
