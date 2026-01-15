"""
Pipeline manager with comprehensive error handling for the News Market Predictor system.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import NewsArticle, MarketPrediction
from .interfaces import (
    NewsFetcher,
    ContentProcessor,
    SentimentAnalyzer,
    EntityExtractor,
    MarketPredictor,
)
from .error_handling import (
    ErrorHandlingManager,
    Priority,
    RetryConfig,
    RateLimitConfig,
    ResourceConstraints,
    with_error_recovery,
)
from .exceptions import NewsMarketPredictorError


logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Manages the complete news analysis pipeline with comprehensive error handling.
    """

    def __init__(
        self,
        fetcher: NewsFetcher,
        processor: ContentProcessor,
        sentiment_analyzer: SentimentAnalyzer,
        entity_extractor: EntityExtractor,
        predictor: MarketPredictor,
        storage=None,
        max_concurrent_tasks: int = 10,
        max_memory_mb: int = 1024,
    ):
        """Initialize pipeline manager with error handling."""
        self.fetcher = fetcher
        self.processor = processor
        self.sentiment_analyzer = sentiment_analyzer
        self.entity_extractor = entity_extractor
        self.predictor = predictor
        self.storage = storage

        # Setup comprehensive error handling
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
        )

        rate_limit_config = RateLimitConfig(
            requests_per_second=2.0, burst_size=5, cooldown_period=60.0
        )

        resource_constraints = ResourceConstraints(
            max_memory_mb=max_memory_mb,
            max_cpu_percent=80.0,
            max_concurrent_tasks=max_concurrent_tasks,
        )

        self.error_manager = ErrorHandlingManager(
            retry_config=retry_config,
            rate_limit_config=rate_limit_config,
            resource_constraints=resource_constraints,
        )

        # Setup storage recovery if storage is provided
        if storage:
            self.error_manager.setup_storage_recovery(storage)

        # Statistics tracking
        self.stats = {
            "articles_processed": 0,
            "articles_failed": 0,
            "predictions_generated": 0,
            "predictions_failed": 0,
            "storage_failures": 0,
            "rate_limit_hits": 0,
        }

    def run_daily_analysis(
        self, target_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Run complete daily news analysis with error handling and resource management.

        Args:
            target_date: Date to analyze (defaults to today)

        Returns:
            Dictionary containing analysis results and statistics
        """
        if target_date is None:
            target_date = datetime.now()

        logger.info("Starting daily analysis for %s", target_date.strftime("%Y-%m-%d"))

        try:
            # Step 1: Fetch news articles with error handling
            articles = self._fetch_news_with_recovery(target_date)
            if not articles:
                logger.warning(
                    "No articles fetched for %s", target_date.strftime("%Y-%m-%d")
                )
                return self._create_result_summary([], [])

            logger.info("Fetched %d articles for processing", len(articles))

            # Step 2: Process articles with resource prioritization
            predictions = self._process_articles_with_priority(articles)

            # Step 3: Store results with backup recovery
            if self.storage:
                self._store_results_with_recovery(predictions)

            logger.info(
                "Daily analysis completed. Generated %d predictions", len(predictions)
            )

            return self._create_result_summary(articles, predictions)

        except Exception as e:
            logger.error("Critical error in daily analysis: %s", e)
            self.stats["articles_failed"] += 1
            return self._create_error_result(str(e))

    @with_error_recovery(fallback_value=[], log_errors=True)
    def _fetch_news_with_recovery(self, target_date: datetime) -> List[NewsArticle]:
        """Fetch news with error recovery."""
        try:
            # Use retry decorator for network operations
            retry_decorator = self.error_manager.get_retry_decorator(
                exceptions=(NewsMarketPredictorError,)
            )

            @retry_decorator
            def fetch_with_retry():
                return self.fetcher.fetch_daily_news(target_date)

            return fetch_with_retry()

        except Exception as e:
            logger.error("Failed to fetch news after retries: %s", e)
            self.stats["articles_failed"] += 1
            return []

    def _process_articles_with_priority(
        self, articles: List[NewsArticle]
    ) -> List[MarketPrediction]:
        """Process articles with resource prioritization and error handling."""
        all_predictions = []

        def process_single_article(article: NewsArticle) -> List[MarketPrediction]:
            """Process a single article through the complete pipeline."""
            try:
                self.stats["articles_processed"] += 1

                # Step 1: Process content
                processed_article = self._process_content_with_recovery(article)
                if not processed_article:
                    return []

                # Step 2: Analyze sentiment
                sentiment = self._analyze_sentiment_with_recovery(processed_article)
                if not sentiment:
                    return []

                # Step 3: Extract entities
                entities = self._extract_entities_with_recovery(processed_article)
                if not entities:
                    return []

                # Step 4: Generate predictions
                predictions = self._generate_predictions_with_recovery(
                    processed_article, sentiment, entities
                )

                self.stats["predictions_generated"] += len(predictions)
                return predictions

            except Exception as e:
                logger.error("Error processing article %s: %s", article.id, e)
                self.stats["articles_failed"] += 1
                return []

        # Process articles with resource prioritization
        try:
            results = self.error_manager.process_with_priority(
                articles, process_single_article
            )
            for result in results:
                if isinstance(result, list):
                    all_predictions.extend(result)
                elif result:
                    all_predictions.append(result)
        except Exception as e:
            logger.error("Error in prioritized processing: %s", e)

        return all_predictions

    @with_error_recovery(fallback_value=None, log_errors=True)
    def _process_content_with_recovery(
        self, article: NewsArticle
    ) -> Optional[NewsArticle]:
        """Process article content with error recovery."""
        try:
            return self.processor.process_content(article)
        except Exception as e:
            logger.warning(
                "Content processing failed for article %s: %s", article.id, e
            )
            return article  # Return original article as fallback

    @with_error_recovery(fallback_value=None, log_errors=True)
    def _analyze_sentiment_with_recovery(self, article: NewsArticle):
        """Analyze sentiment with error recovery."""
        try:
            return self.sentiment_analyzer.analyze_sentiment(article)
        except Exception as e:
            logger.warning(
                "Sentiment analysis failed for article %s: %s", article.id, e
            )
            # Return neutral sentiment as fallback
            from .models import SentimentAnalysis

            return SentimentAnalysis(
                article_id=article.id,
                sentiment_score=0.0,
                confidence=0.0,
                key_phrases=[],
                market_tone="neutral",
            )

    @with_error_recovery(fallback_value=[], log_errors=True)
    def _extract_entities_with_recovery(self, article: NewsArticle):
        """Extract entities with error recovery."""
        try:
            return self.entity_extractor.extract_entities(article)
        except Exception as e:
            logger.warning("Entity extraction failed for article %s: %s", article.id, e)
            return []

    @with_error_recovery(fallback_value=[], log_errors=True)
    def _generate_predictions_with_recovery(
        self, article: NewsArticle, sentiment, entities
    ):
        """Generate predictions with error recovery."""
        try:
            predictions = self.predictor.predict_impact(article, sentiment, entities)
            return predictions
        except Exception as e:
            logger.warning(
                "Prediction generation failed for article %s: %s", article.id, e
            )
            self.stats["predictions_failed"] += 1
            return []

    def _store_results_with_recovery(self, predictions: List[MarketPrediction]) -> None:
        """Store results with backup recovery."""
        if not self.storage or not predictions:
            return

        successful_stores = 0
        failed_stores = 0

        for prediction in predictions:
            try:
                if self.storage.store_prediction(prediction):
                    successful_stores += 1
                else:
                    failed_stores += 1
                    self.stats["storage_failures"] += 1
            except Exception as e:
                logger.error(
                    "Storage error for prediction %s: %s", prediction.article_id, e
                )
                failed_stores += 1
                self.stats["storage_failures"] += 1

        logger.info(
            "Storage results: %d successful, %d failed",
            successful_stores,
            failed_stores,
        )

    def _create_result_summary(
        self, articles: List[NewsArticle], predictions: List[MarketPrediction]
    ) -> Dict[str, Any]:
        """Create summary of analysis results."""
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "articles_count": len(articles),
            "predictions_count": len(predictions),
            "statistics": self.stats.copy(),
            "predictions": [
                {
                    "stock_symbol": p.stock_symbol,
                    "impact_direction": p.impact_direction,
                    "confidence_level": p.confidence_level,
                    "reasoning": p.reasoning,
                }
                for p in predictions[:10]  # Limit to first 10 for summary
            ],
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result summary."""
        return {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "statistics": self.stats.copy(),
            "articles_count": 0,
            "predictions_count": 0,
            "predictions": [],
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status including error rates."""
        total_articles = (
            self.stats["articles_processed"] + self.stats["articles_failed"]
        )
        total_predictions = (
            self.stats["predictions_generated"] + self.stats["predictions_failed"]
        )

        article_success_rate = (
            self.stats["articles_processed"] / total_articles
            if total_articles > 0
            else 1.0
        )

        prediction_success_rate = (
            self.stats["predictions_generated"] / total_predictions
            if total_predictions > 0
            else 1.0
        )

        return {
            "status": (
                "healthy"
                if article_success_rate > 0.8 and prediction_success_rate > 0.8
                else "degraded"
            ),
            "article_success_rate": article_success_rate,
            "prediction_success_rate": prediction_success_rate,
            "storage_failure_count": self.stats["storage_failures"],
            "rate_limit_hits": self.stats["rate_limit_hits"],
            "statistics": self.stats.copy(),
        }

    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            "articles_processed": 0,
            "articles_failed": 0,
            "predictions_generated": 0,
            "predictions_failed": 0,
            "storage_failures": 0,
            "rate_limit_hits": 0,
        }
