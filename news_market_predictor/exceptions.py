"""
Custom exceptions for the News Market Predictor system.
"""


class NewsMarketPredictorError(Exception):
    """Base exception for all News Market Predictor errors."""

    pass


class NewsFetchError(NewsMarketPredictorError):
    """Raised when news fetching operations fail."""

    pass


class NetworkError(NewsFetchError):
    """Raised when network operations fail."""

    pass


class RateLimitError(NewsFetchError):
    """Raised when API rate limits are exceeded."""

    pass


class AccessDeniedError(NewsFetchError):
    """Raised when access to a resource is denied (401/403 errors)."""

    pass


class ParsingError(NewsFetchError):
    """Raised when content parsing operations fail."""

    pass


class ContentProcessingError(NewsMarketPredictorError):
    """Raised when content processing operations fail."""

    pass


class SentimentAnalysisError(NewsMarketPredictorError):
    """Raised when sentiment analysis operations fail."""

    pass


class EntityExtractionError(NewsMarketPredictorError):
    """Raised when entity extraction operations fail."""

    pass


class PredictionError(NewsMarketPredictorError):
    """Raised when prediction generation fails."""

    pass


class StorageError(NewsMarketPredictorError):
    """Raised when data storage operations fail."""

    pass


class ValidationError(NewsMarketPredictorError):
    """Raised when data validation fails."""

    pass


class ConfigurationError(NewsMarketPredictorError):
    """Raised when configuration is invalid or missing."""

    pass
