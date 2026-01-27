"""
LLM service specific exceptions.
"""


class LLMServiceError(Exception):
    """Base exception for LLM service errors."""

    pass


class LLMProviderError(LLMServiceError):
    """Exception raised when an LLM provider encounters an error."""

    def __init__(self, message: str, provider: str = None, error_code: str = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code


class LLMRateLimitError(LLMProviderError):
    """Exception raised when rate limits are exceeded."""

    def __init__(self, message: str, provider: str = None, retry_after: int = None):
        super().__init__(message, provider, "RATE_LIMIT")
        self.retry_after = retry_after


class LLMTimeoutError(LLMProviderError):
    """Exception raised when LLM requests timeout."""

    def __init__(self, message: str, provider: str = None, timeout_seconds: int = None):
        super().__init__(message, provider, "TIMEOUT")
        self.timeout_seconds = timeout_seconds


class LLMInvalidResponseError(LLMProviderError):
    """Exception raised when LLM response is invalid or malformed."""

    def __init__(self, message: str, provider: str = None, response_data: str = None):
        super().__init__(message, provider, "INVALID_RESPONSE")
        self.response_data = response_data


class LLMConfigurationError(LLMServiceError):
    """Exception raised when LLM configuration is invalid."""

    pass


class LLMBudgetExceededError(LLMServiceError):
    """Exception raised when budget limits are exceeded."""

    def __init__(
        self, message: str, current_cost: float = None, budget_limit: float = None
    ):
        super().__init__(message)
        self.current_cost = current_cost
        self.budget_limit = budget_limit
