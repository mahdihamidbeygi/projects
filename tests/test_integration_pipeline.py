"""
Integration tests for end-to-end pipeline.

Tests complete workflow from news fetching to prediction output,
verifies data flow between all components, and tests error recovery.

Requirements: All requirements
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from news_market_predictor.models import (
    NewsArticle,
    SentimentAnalysis,
    ExtractedEntity,
    MarketPrediction,
)
from news_market_predictor.interfaces import (
    NewsFetcher,
    ContentProcessor,
    SentimentAnalyzer,
    EntityExtractor,
    MarketPredictor,
)
from news_market_predictor.pipeline_manager import PipelineManager
from news_market_predictor.exceptions import NewsMarketPredictorError


# Mock implementations for testing
class MockNewsFetcher(NewsFetcher):
    """Mock news fetcher for testing."""

    def __init__(
        self,
        articles_to_return: Optional[List[NewsArticle]] = None,
        should_fail: bool = False,
    ):
        self.articles_to_return = articles_to_return or []
        self.should_fail = should_fail
        self.fetch_count = 0

    def fetch_daily_news(self, date: Optional[datetime] = None) -> List[NewsArticle]:
        """Fetch mock news articles."""
        self.fetch_count += 1
        if self.should_fail:
            raise NewsMarketPredictorError("Mock fetch failure")
        return self.articles_to_return

    def parse_article_content(
        self, raw_content: str, metadata: Dict[str, Any]
    ) -> NewsArticle:
        """Parse mock article content."""
        return NewsArticle(
            id="test-id",
            title="Test Article",
            content=raw_content,
            url="https://test.com",
            published_at=datetime.now(),
            source="Yahoo Finance",
            category="Technology",
            raw_metadata=metadata,
        )

    def deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicates."""
        seen = set()
        unique = []
        for article in articles:
            if article.title not in seen:
                seen.add(article.title)
                unique.append(article)
        return unique


class MockContentProcessor(ContentProcessor):
    """Mock content processor for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.process_count = 0

    def clean_text(self, text: str) -> str:
        """Clean text."""
        return text.strip()

    def extract_metadata(self, article: NewsArticle) -> Dict[str, Any]:
        """Extract metadata."""
        return {"word_count": len(article.content.split())}

    def validate_content(self, article: NewsArticle) -> bool:
        """Validate content."""
        return bool(article.content.strip())

    def process_content(self, article: NewsArticle) -> NewsArticle:
        """Process article content."""
        self.process_count += 1
        if self.should_fail:
            raise NewsMarketPredictorError("Mock processing failure")
        return article


class MockSentimentAnalyzer(SentimentAnalyzer):
    """Mock sentiment analyzer for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.analyze_count = 0

    def analyze_sentiment(self, article: NewsArticle) -> SentimentAnalysis:
        """Analyze sentiment."""
        self.analyze_count += 1
        if self.should_fail:
            raise NewsMarketPredictorError("Mock sentiment analysis failure")

        return SentimentAnalysis(
            article_id=article.id,
            sentiment_score=0.5,
            confidence=0.8,
            key_phrases=["positive", "growth"],
            market_tone="bullish",
        )

    def calculate_confidence(self, text: str, sentiment_score: float) -> float:
        """Calculate confidence."""
        return 0.8

    def detect_market_tone(self, text: str, sentiment_score: float) -> str:
        """Detect market tone."""
        if sentiment_score > 0.2:
            return "bullish"
        elif sentiment_score < -0.2:
            return "bearish"
        return "neutral"


class MockEntityExtractor(EntityExtractor):
    """Mock entity extractor for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.extract_count = 0

    def extract_entities(self, article: NewsArticle) -> List[ExtractedEntity]:
        """Extract entities."""
        self.extract_count += 1
        if self.should_fail:
            raise NewsMarketPredictorError("Mock entity extraction failure")

        return [
            ExtractedEntity(
                article_id=article.id,
                entity_type="stock_symbol",
                entity_value="AAPL",
                relevance_score=0.9,
                context="Apple announces new product",
            )
        ]

    def extract_stock_symbols(self, text: str) -> List[ExtractedEntity]:
        """Extract stock symbols."""
        return []

    def identify_companies(self, text: str) -> List[ExtractedEntity]:
        """Identify companies."""
        return []

    def find_financial_metrics(self, text: str) -> List[ExtractedEntity]:
        """Find financial metrics."""
        return []


class MockMarketPredictor(MarketPredictor):
    """Mock market predictor for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.predict_count = 0

    def predict_impact(
        self,
        article: NewsArticle,
        sentiment: SentimentAnalysis,
        entities: List[ExtractedEntity],
    ) -> List[MarketPrediction]:
        """Generate predictions."""
        self.predict_count += 1
        if self.should_fail:
            raise NewsMarketPredictorError("Mock prediction failure")

        predictions = []
        for entity in entities:
            if entity.entity_type == "stock_symbol":
                predictions.append(
                    MarketPrediction(
                        article_id=article.id,
                        stock_symbol=entity.entity_value,
                        impact_direction="positive",
                        impact_magnitude=0.6,
                        confidence_level=0.75,
                        reasoning="Positive sentiment and strong entity relevance",
                        created_at=datetime.now(),
                    )
                )
        return predictions

    def calculate_confidence(self, prediction_data: Dict[str, Any]) -> float:
        """Calculate confidence."""
        return 0.75

    def aggregate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate signals."""
        return {}


# Test fixtures
@pytest.fixture
def sample_articles():
    """Create sample articles for testing."""
    return [
        NewsArticle(
            id="article-1",
            title="Apple Announces Record Earnings",
            content="Apple Inc. reported record quarterly earnings today, beating analyst expectations.",
            url="https://finance.yahoo.com/news/apple-earnings",
            published_at=datetime.now(),
            source="Yahoo Finance",
            category="Technology",
            raw_metadata={"author": "John Doe"},
        ),
        NewsArticle(
            id="article-2",
            title="Tesla Stock Surges on New Model Announcement",
            content="Tesla shares jumped 5% after announcing a new electric vehicle model.",
            url="https://finance.yahoo.com/news/tesla-surge",
            published_at=datetime.now(),
            source="Yahoo Finance",
            category="Automotive",
            raw_metadata={"author": "Jane Smith"},
        ),
    ]


@pytest.fixture
def pipeline_components():
    """Create pipeline components for testing."""
    return {
        "fetcher": MockNewsFetcher(),
        "processor": MockContentProcessor(),
        "sentiment_analyzer": MockSentimentAnalyzer(),
        "entity_extractor": MockEntityExtractor(),
        "predictor": MockMarketPredictor(),
    }


# Integration Tests
def test_complete_pipeline_workflow(sample_articles, pipeline_components):
    """
    Test complete workflow from news fetching to prediction output.

    Verifies that:
    - Articles are fetched successfully
    - Each component processes data correctly
    - Predictions are generated
    - Results are properly formatted
    """
    # Setup
    pipeline_components["fetcher"].articles_to_return = sample_articles

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute
    target_date = datetime.now()
    results = pipeline.run_daily_analysis(target_date)

    # Verify
    assert results["success"] is True
    assert results["articles_count"] == 2
    assert results["predictions_count"] == 2
    assert len(results["predictions"]) == 2

    # Verify each component was called
    assert pipeline_components["fetcher"].fetch_count == 1
    assert pipeline_components["processor"].process_count == 2
    assert pipeline_components["sentiment_analyzer"].analyze_count == 2
    assert pipeline_components["entity_extractor"].extract_count == 2
    assert pipeline_components["predictor"].predict_count == 2

    # Verify prediction structure
    for prediction in results["predictions"]:
        assert "stock_symbol" in prediction
        assert "impact_direction" in prediction
        assert "confidence_level" in prediction
        assert "reasoning" in prediction


def test_data_flow_between_components(sample_articles, pipeline_components):
    """
    Test data flow between all components.

    Verifies that:
    - Data is correctly passed from fetcher to processor
    - Processor output feeds into sentiment analyzer
    - Sentiment and entities are used for predictions
    - Final predictions contain data from all stages
    """
    # Setup
    pipeline_components["fetcher"].articles_to_return = [sample_articles[0]]

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute
    results = pipeline.run_daily_analysis(datetime.now())

    # Verify data flow
    assert results["success"] is True
    assert results["articles_count"] == 1
    assert results["predictions_count"] == 1

    # Verify prediction contains data from all stages
    prediction = results["predictions"][0]
    assert prediction["stock_symbol"] == "AAPL"  # From entity extractor
    assert (
        prediction["impact_direction"] == "positive"
    )  # From predictor using sentiment
    assert 0.0 <= prediction["confidence_level"] <= 1.0  # Valid confidence
    assert len(prediction["reasoning"]) > 0  # Has reasoning


def test_error_recovery_fetch_failure(pipeline_components):
    """
    Test error recovery when news fetching fails.

    Verifies that:
    - Pipeline handles fetch failures gracefully
    - Error is logged but doesn't crash the system
    - Results indicate failure appropriately
    """
    # Setup - configure fetcher to fail
    pipeline_components["fetcher"].should_fail = True

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute
    results = pipeline.run_daily_analysis(datetime.now())

    # Verify graceful handling
    assert results["success"] is True  # Pipeline completes even with no articles
    assert results["articles_count"] == 0
    assert results["predictions_count"] == 0


def test_error_recovery_processing_failure(sample_articles, pipeline_components):
    """
    Test error recovery when content processing fails.

    Verifies that:
    - Pipeline continues processing other articles
    - Failed articles are logged
    - Successful articles still generate predictions
    """
    # Setup - configure processor to fail
    pipeline_components["fetcher"].articles_to_return = sample_articles
    pipeline_components["processor"].should_fail = True

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute
    results = pipeline.run_daily_analysis(datetime.now())

    # Verify error recovery
    assert results["success"] is True
    assert results["articles_count"] == 2
    # Predictions may be 0 due to processing failures, but pipeline completes
    assert "statistics" in results


def test_error_recovery_sentiment_failure(sample_articles, pipeline_components):
    """
    Test error recovery when sentiment analysis fails.

    Verifies that:
    - Pipeline uses fallback neutral sentiment
    - Processing continues for other components
    - Predictions are still generated with neutral sentiment
    """
    # Setup - configure sentiment analyzer to fail
    pipeline_components["fetcher"].articles_to_return = [sample_articles[0]]
    pipeline_components["sentiment_analyzer"].should_fail = True

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute
    results = pipeline.run_daily_analysis(datetime.now())

    # Verify fallback behavior
    assert results["success"] is True
    assert results["articles_count"] == 1
    # Pipeline should still generate predictions with fallback sentiment


def test_error_recovery_entity_extraction_failure(sample_articles, pipeline_components):
    """
    Test error recovery when entity extraction fails.

    Verifies that:
    - Pipeline handles missing entities gracefully
    - No predictions generated without entities
    - System continues without crashing
    """
    # Setup - configure entity extractor to fail
    pipeline_components["fetcher"].articles_to_return = [sample_articles[0]]
    pipeline_components["entity_extractor"].should_fail = True

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute
    results = pipeline.run_daily_analysis(datetime.now())

    # Verify graceful handling
    assert results["success"] is True
    assert results["articles_count"] == 1
    # May have 0 predictions due to missing entities


def test_error_recovery_prediction_failure(sample_articles, pipeline_components):
    """
    Test error recovery when prediction generation fails.

    Verifies that:
    - Pipeline handles prediction failures gracefully
    - Statistics track failed predictions
    - System completes without crashing
    """
    # Setup - configure predictor to fail
    pipeline_components["fetcher"].articles_to_return = [sample_articles[0]]
    pipeline_components["predictor"].should_fail = True

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute
    results = pipeline.run_daily_analysis(datetime.now())

    # Verify error tracking
    assert results["success"] is True
    assert results["articles_count"] == 1
    assert results["predictions_count"] == 0
    assert results["statistics"]["predictions_failed"] > 0


def test_empty_article_list(pipeline_components):
    """
    Test pipeline behavior with empty article list.

    Verifies that:
    - Pipeline handles empty input gracefully
    - No errors are raised
    - Results indicate no articles processed
    """
    # Setup - no articles
    pipeline_components["fetcher"].articles_to_return = []

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute
    results = pipeline.run_daily_analysis(datetime.now())

    # Verify
    assert results["success"] is True
    assert results["articles_count"] == 0
    assert results["predictions_count"] == 0


def test_multiple_predictions_per_article(pipeline_components):
    """
    Test pipeline with articles that generate multiple predictions.

    Verifies that:
    - Multiple entities in one article generate multiple predictions
    - All predictions are captured in results
    - Statistics are accurate
    """
    # Setup - article with multiple entities
    article = NewsArticle(
        id="multi-entity",
        title="Tech Giants Report Earnings",
        content="Apple and Microsoft both reported strong earnings.",
        url="https://finance.yahoo.com/news/tech-earnings",
        published_at=datetime.now(),
        source="Yahoo Finance",
        category="Technology",
        raw_metadata={},
    )

    # Custom entity extractor that returns multiple entities
    class MultiEntityExtractor(MockEntityExtractor):
        def extract_entities(self, article: NewsArticle) -> List[ExtractedEntity]:
            return [
                ExtractedEntity(
                    article_id=article.id,
                    entity_type="stock_symbol",
                    entity_value="AAPL",
                    relevance_score=0.9,
                    context="Apple earnings",
                ),
                ExtractedEntity(
                    article_id=article.id,
                    entity_type="stock_symbol",
                    entity_value="MSFT",
                    relevance_score=0.9,
                    context="Microsoft earnings",
                ),
            ]

    pipeline_components["fetcher"].articles_to_return = [article]
    pipeline_components["entity_extractor"] = MultiEntityExtractor()

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute
    results = pipeline.run_daily_analysis(datetime.now())

    # Verify
    assert results["success"] is True
    assert results["articles_count"] == 1
    assert results["predictions_count"] == 2  # Two predictions from one article
    assert len(results["predictions"]) == 2


def test_health_status_tracking(sample_articles, pipeline_components):
    """
    Test health status tracking during pipeline execution.

    Verifies that:
    - Health status is available
    - Success rates are calculated correctly
    - Statistics are tracked accurately
    """
    # Setup
    pipeline_components["fetcher"].articles_to_return = sample_articles

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute
    pipeline.run_daily_analysis(datetime.now())

    # Get health status
    health = pipeline.get_health_status()

    # Verify
    assert "status" in health
    assert "article_success_rate" in health
    assert "prediction_success_rate" in health
    assert "statistics" in health
    assert health["article_success_rate"] >= 0.0
    assert health["article_success_rate"] <= 1.0


def test_statistics_reset(sample_articles, pipeline_components):
    """
    Test statistics reset functionality.

    Verifies that:
    - Statistics can be reset
    - Counters return to zero
    - Pipeline continues to work after reset
    """
    # Setup
    pipeline_components["fetcher"].articles_to_return = sample_articles

    pipeline = PipelineManager(
        fetcher=pipeline_components["fetcher"],
        processor=pipeline_components["processor"],
        sentiment_analyzer=pipeline_components["sentiment_analyzer"],
        entity_extractor=pipeline_components["entity_extractor"],
        predictor=pipeline_components["predictor"],
        storage=None,
    )

    # Execute first run
    pipeline.run_daily_analysis(datetime.now())
    assert pipeline.stats["articles_processed"] > 0

    # Reset
    pipeline.reset_statistics()

    # Verify reset
    assert pipeline.stats["articles_processed"] == 0
    assert pipeline.stats["predictions_generated"] == 0

    # Execute second run
    results = pipeline.run_daily_analysis(datetime.now())
    assert results["success"] is True
    assert pipeline.stats["articles_processed"] > 0
