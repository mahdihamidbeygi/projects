"""
Comprehensive error handling utilities for the News Market Predictor system.
"""

import time
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .exceptions import (
    RateLimitError,
    StorageError,
    PredictionError,
    NetworkError,
    NewsMarketPredictorError,
)


logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority levels for resource allocation."""

    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 1.0
    burst_size: int = 5
    cooldown_period: float = 60.0


@dataclass
class ResourceConstraints:
    """System resource constraints."""

    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    max_concurrent_tasks: int = 10


class RateLimitHandler:
    """Handles API rate limiting with adaptive delays."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times: List[float] = []
        self.last_rate_limit_time: Optional[float] = None
        self.current_delay = 1.0

    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        current_time = time.time()

        # Clean old request times (older than 1 second)
        cutoff_time = current_time - 1.0
        self.request_times = [t for t in self.request_times if t > cutoff_time]

        # Check if we're within rate limits
        if len(self.request_times) >= self.config.requests_per_second:
            # Calculate wait time
            oldest_request = min(self.request_times)
            wait_time = 1.0 - (current_time - oldest_request)

            if wait_time > 0:
                logger.info("Rate limit reached, waiting %.2f seconds", wait_time)
                time.sleep(wait_time)

        # Record this request
        self.request_times.append(time.time())

    def handle_rate_limit_response(self, retry_after: Optional[int] = None) -> None:
        """Handle rate limit response from API."""
        self.last_rate_limit_time = time.time()

        if retry_after:
            wait_time = min(retry_after, self.config.cooldown_period)
        else:
            # Exponential backoff
            self.current_delay = min(
                self.current_delay * 2, self.config.cooldown_period
            )
            wait_time = self.current_delay

        logger.warning("Rate limited by server, waiting %.2f seconds", wait_time)
        time.sleep(wait_time)

    def reset_delay(self) -> None:
        """Reset delay after successful requests."""
        self.current_delay = 1.0


class StorageFailureRecovery:
    """Handles storage failures with backup systems."""

    def __init__(self, primary_storage, backup_storage=None):
        self.primary_storage = primary_storage
        self.backup_storage = backup_storage
        self.primary_failed = False
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None

    def store_with_recovery(self, operation: str, *args, **kwargs) -> bool:
        """Attempt storage operation with fallback to backup."""
        # If primary is marked as failed, skip directly to backup
        if self.primary_failed:
            if self.backup_storage:
                try:
                    method = getattr(self.backup_storage, operation)
                    result = method(*args, **kwargs)
                    if result:
                        logger.info(
                            "Using backup storage for %s (primary marked as failed)",
                            operation,
                        )
                        return True
                except Exception as backup_error:
                    logger.error(
                        "Backup storage failed for %s: %s", operation, backup_error
                    )

            # No backup or backup failed
            self._alert_administrators(
                f"Storage operation {operation} failed (primary unavailable)"
            )
            return False

        try:
            # Try primary storage first
            method = getattr(self.primary_storage, operation)
            result = method(*args, **kwargs)
            if result:
                # Reset failure state on success
                self.primary_failed = False
                self.failure_count = 0
                return True
            else:
                raise StorageError(f"Primary storage {operation} returned False")

        except Exception as e:
            logger.error("Primary storage failed for %s: %s", operation, e)
            self._handle_primary_failure()

            # Try backup storage if available
            if self.backup_storage:
                try:
                    method = getattr(self.backup_storage, operation)
                    result = method(*args, **kwargs)
                    if result:
                        logger.info(
                            "Successfully used backup storage for %s", operation
                        )
                        self._alert_administrators(
                            f"Primary storage failed, using backup for {operation}"
                        )
                        return True
                except Exception as backup_error:
                    logger.error(
                        "Backup storage also failed for %s: %s", operation, backup_error
                    )

            # Both storages failed
            self._alert_administrators(f"All storage systems failed for {operation}")
            return False

    def _handle_primary_failure(self) -> None:
        """Handle primary storage failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        # Mark primary as failed if multiple failures
        if self.failure_count >= 3:
            self.primary_failed = True
            logger.warning(
                "Primary storage marked as failed after %d failures", self.failure_count
            )

    def _alert_administrators(self, message: str) -> None:
        """Send alert to administrators (placeholder implementation)."""
        logger.critical("ADMIN ALERT: %s", message)
        # In a real implementation, this would send emails, Slack messages, etc.

    def check_primary_recovery(self) -> None:
        """Check if primary storage has recovered."""
        if self.primary_failed and self.last_failure_time:
            # Try to recover after 5 minutes
            if datetime.now() - self.last_failure_time > timedelta(minutes=5):
                try:
                    # Test primary storage with a simple operation
                    if hasattr(self.primary_storage, "get_database_stats"):
                        self.primary_storage.get_database_stats()
                        self.primary_failed = False
                        self.failure_count = 0
                        logger.info("Primary storage has recovered")
                except Exception:
                    logger.debug("Primary storage still not available")


class ResourcePrioritizer:
    """Manages resource allocation based on priority."""

    def __init__(self, constraints: ResourceConstraints):
        self.constraints = constraints
        self.active_tasks: Dict[Priority, int] = {
            Priority.HIGH: 0,
            Priority.MEDIUM: 0,
            Priority.LOW: 0,
        }
        self.task_queue: List[tuple] = []  # (priority, task_data)

    def can_process_task(self, priority: Priority) -> bool:
        """Check if a task can be processed given current resource usage."""
        total_tasks = sum(self.active_tasks.values())

        # Always allow high priority tasks if under absolute limit
        if priority == Priority.HIGH:
            return total_tasks < self.constraints.max_concurrent_tasks

        # For medium priority, ensure high priority tasks have resources
        if priority == Priority.MEDIUM:
            high_priority_slots = max(2, self.constraints.max_concurrent_tasks // 4)
            available_slots = (
                self.constraints.max_concurrent_tasks - high_priority_slots
            )
            medium_and_low_tasks = (
                self.active_tasks[Priority.MEDIUM] + self.active_tasks[Priority.LOW]
            )
            return medium_and_low_tasks < available_slots

        # For low priority, only use remaining resources
        if priority == Priority.LOW:
            reserved_slots = max(4, self.constraints.max_concurrent_tasks // 2)
            available_slots = self.constraints.max_concurrent_tasks - reserved_slots
            return self.active_tasks[Priority.LOW] < available_slots

        return False

    def start_task(self, priority: Priority) -> bool:
        """Mark a task as started if resources allow."""
        if self.can_process_task(priority):
            self.active_tasks[priority] += 1
            return True
        return False

    def finish_task(self, priority: Priority) -> None:
        """Mark a task as finished."""
        if self.active_tasks[priority] > 0:
            self.active_tasks[priority] -= 1

    def get_processing_order(self, articles: List[Any]) -> List[Any]:
        """Sort articles by processing priority based on content analysis."""

        def get_priority(article) -> Priority:
            # Analyze article for high-impact indicators
            content = getattr(article, "content", "").lower()
            title = getattr(article, "title", "").lower()

            high_impact_keywords = [
                "earnings",
                "acquisition",
                "merger",
                "bankruptcy",
                "lawsuit",
                "fda approval",
                "clinical trial",
                "breakthrough",
                "partnership",
            ]

            medium_impact_keywords = [
                "revenue",
                "profit",
                "guidance",
                "upgrade",
                "downgrade",
                "analyst",
                "rating",
                "target price",
            ]

            text = f"{title} {content}"

            if any(keyword in text for keyword in high_impact_keywords):
                return Priority.HIGH
            elif any(keyword in text for keyword in medium_impact_keywords):
                return Priority.MEDIUM
            else:
                return Priority.LOW

        # Sort by priority (HIGH=1, MEDIUM=2, LOW=3)
        return sorted(articles, key=lambda x: get_priority(x).value)


class InvalidInputHandler:
    """Handles invalid inputs to prediction models."""

    @staticmethod
    def validate_article(article) -> bool:
        """Validate article input."""
        if not article:
            return False

        # Check required fields exist and are not None
        required_fields = ["title", "content", "published_at"]
        for field in required_fields:
            if not hasattr(article, field):
                return False
            value = getattr(article, field)
            if value is None:
                return False
            # For string fields, check they're not empty
            if field in ["title", "content"]:
                if not isinstance(value, str) or len(value.strip()) == 0:
                    return False

        # Check minimum content length (at least 10 characters after stripping)
        content = getattr(article, "content", "")
        if len(content.strip()) < 10:
            return False

        return True

    @staticmethod
    def validate_sentiment(sentiment) -> bool:
        """Validate sentiment analysis input."""
        if not sentiment:
            return False

        # Check sentiment score bounds
        score = getattr(sentiment, "sentiment_score", None)
        if score is None or not (-1.0 <= score <= 1.0):
            return False

        # Check confidence bounds
        confidence = getattr(sentiment, "confidence", None)
        if confidence is None or not (0.0 <= confidence <= 1.0):
            return False

        return True

    @staticmethod
    def validate_entities(entities) -> bool:
        """Validate extracted entities input."""
        if not isinstance(entities, list):
            return False

        for entity in entities:
            if not hasattr(entity, "entity_type") or not hasattr(
                entity, "entity_value"
            ):
                return False

            # Check that entity_type and entity_value are not None
            if getattr(entity, "entity_type", None) is None:
                return False
            if getattr(entity, "entity_value", None) is None:
                return False

            # Check relevance score bounds
            relevance = getattr(entity, "relevance_score", None)
            if relevance is not None and not (0.0 <= relevance <= 1.0):
                return False

        return True

    @staticmethod
    def create_neutral_prediction(
        article_id: str, stock_symbol: str, error_message: str
    ):
        """Create a neutral prediction for invalid inputs."""
        from .models import MarketPrediction

        return MarketPrediction(
            article_id=article_id,
            stock_symbol=stock_symbol,
            impact_direction="neutral",
            impact_magnitude=0.0,
            confidence_level=0.0,
            reasoning=f"Invalid input detected: {error_message}",
            created_at=datetime.now(),
        )


def with_retry(
    retry_config: RetryConfig,
    exceptions: tuple = (Exception,),
    rate_limit_handler: Optional[RateLimitHandler] = None,
):
    """Decorator for adding retry logic with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(retry_config.max_attempts):
                try:
                    # Apply rate limiting if handler provided
                    if rate_limit_handler:
                        rate_limit_handler.wait_if_needed()

                    result = func(*args, **kwargs)

                    # Reset rate limit delay on success
                    if rate_limit_handler:
                        rate_limit_handler.reset_delay()

                    return result

                except RateLimitError as e:
                    if rate_limit_handler:
                        retry_after = getattr(e, "retry_after", None)
                        rate_limit_handler.handle_rate_limit_response(retry_after)
                    last_exception = e

                except exceptions as e:
                    last_exception = e

                    if attempt < retry_config.max_attempts - 1:
                        # Calculate delay with exponential backoff
                        delay = min(
                            retry_config.base_delay
                            * (retry_config.exponential_base**attempt),
                            retry_config.max_delay,
                        )

                        # Add jitter to prevent thundering herd
                        if retry_config.jitter:
                            import random

                            delay *= 0.5 + random.random() * 0.5

                        logger.warning(
                            "Attempt %d failed for %s: %s. Retrying in %.2f seconds",
                            attempt + 1,
                            func.__name__,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            retry_config.max_attempts,
                            func.__name__,
                            e,
                        )

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator


def with_error_recovery(
    fallback_value: Any = None,
    log_errors: bool = True,
    exceptions: tuple = (Exception,),
):
    """Decorator for graceful error recovery."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_errors:
                    logger.error("Error in %s: %s", func.__name__, e)
                return fallback_value

        return wrapper

    return decorator


class ErrorHandlingManager:
    """Central manager for all error handling components."""

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        resource_constraints: Optional[ResourceConstraints] = None,
    ):
        self.retry_config = retry_config or RetryConfig()
        self.rate_limit_handler = RateLimitHandler(
            rate_limit_config or RateLimitConfig()
        )
        self.resource_prioritizer = ResourcePrioritizer(
            resource_constraints or ResourceConstraints()
        )
        self.storage_recovery: Optional[StorageFailureRecovery] = None
        self.invalid_input_handler = InvalidInputHandler()

    def setup_storage_recovery(self, primary_storage, backup_storage=None):
        """Setup storage failure recovery."""
        self.storage_recovery = StorageFailureRecovery(primary_storage, backup_storage)

    def get_retry_decorator(self, exceptions: tuple = (Exception,)):
        """Get configured retry decorator."""
        return with_retry(self.retry_config, exceptions, self.rate_limit_handler)

    def process_with_priority(self, articles: List[Any], processor_func: Callable):
        """Process articles with resource prioritization."""
        ordered_articles = self.resource_prioritizer.get_processing_order(articles)
        results = []

        for article in ordered_articles:
            # Determine priority
            priority = self._get_article_priority(article)

            # Wait for resources if needed
            while not self.resource_prioritizer.start_task(priority):
                logger.info(
                    "Waiting for resources to process %s priority task", priority.name
                )
                time.sleep(1.0)

            try:
                result = processor_func(article)
                results.append(result)
            finally:
                self.resource_prioritizer.finish_task(priority)

        return results

    def _get_article_priority(self, article) -> Priority:
        """Determine article processing priority."""
        content = getattr(article, "content", "").lower()
        title = getattr(article, "title", "").lower()

        high_impact_keywords = [
            "earnings",
            "acquisition",
            "merger",
            "bankruptcy",
            "lawsuit",
            "fda approval",
            "clinical trial",
            "breakthrough",
            "partnership",
        ]

        medium_impact_keywords = [
            "revenue",
            "profit",
            "guidance",
            "upgrade",
            "downgrade",
            "analyst",
            "rating",
            "target price",
        ]

        text = f"{title} {content}"

        if any(keyword in text for keyword in high_impact_keywords):
            return Priority.HIGH
        elif any(keyword in text for keyword in medium_impact_keywords):
            return Priority.MEDIUM
        else:
            return Priority.LOW
