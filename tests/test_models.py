"""
Tests for core data models.
"""

import json
import pytest
from datetime import datetime

from news_market_predictor.models import (
    NewsArticle,
    SentimentAnalysis,
    ExtractedEntity,
    MarketPrediction,
    ValidationError,
    export_to_csv,
    export_to_json,
)


def test_news_article_creation():
    """Test NewsArticle model creation."""
    article = NewsArticle(
        id="test-123",
        title="Test Article",
        content="This is test content",
        url="https://example.com/test",
        published_at=datetime.now(),
        source="Yahoo Finance",
        category="Technology",
        raw_metadata={"test": "data"},
    )

    assert article.id == "test-123"
    assert article.title == "Test Article"
    assert article.source == "Yahoo Finance"


def test_news_article_validation():
    """Test NewsArticle validation."""
    # Valid article should pass validation
    article = NewsArticle(
        id="test-123",
        title="Test Article",
        content="This is test content",
        url="https://example.com/test",
        published_at=datetime.now(),
        source="Yahoo Finance",
        category="Technology",
        raw_metadata={"test": "data"},
    )
    assert article.validate() is True

    # Invalid article should raise ValidationError
    with pytest.raises(ValidationError):
        invalid_article = NewsArticle(
            id="",  # Empty ID should fail
            title="Test Article",
            content="This is test content",
            url="https://example.com/test",
            published_at=datetime.now(),
            source="Yahoo Finance",
            category="Technology",
            raw_metadata={"test": "data"},
        )
        invalid_article.validate()


def test_news_article_json_serialization():
    """Test NewsArticle JSON serialization and deserialization."""
    original_article = NewsArticle(
        id="test-123",
        title="Test Article",
        content="This is test content",
        url="https://example.com/test",
        published_at=datetime(2023, 1, 1, 12, 0, 0),
        source="Yahoo Finance",
        category="Technology",
        raw_metadata={"test": "data"},
    )

    # Serialize to JSON
    json_str = original_article.to_json()
    assert isinstance(json_str, str)

    # Deserialize from JSON
    restored_article = NewsArticle.from_json(json_str)

    # Check that all fields match
    assert restored_article.id == original_article.id
    assert restored_article.title == original_article.title
    assert restored_article.content == original_article.content
    assert restored_article.url == original_article.url
    assert restored_article.published_at == original_article.published_at
    assert restored_article.source == original_article.source
    assert restored_article.category == original_article.category
    assert restored_article.raw_metadata == original_article.raw_metadata


def test_news_article_csv_serialization():
    """Test NewsArticle CSV serialization and deserialization."""
    original_article = NewsArticle(
        id="test-123",
        title="Test Article",
        content="This is test content",
        url="https://example.com/test",
        published_at=datetime(2023, 1, 1, 12, 0, 0),
        source="Yahoo Finance",
        category="Technology",
        raw_metadata={"test": "data"},
    )

    # Convert to CSV row
    csv_row = original_article.to_csv_row()
    assert isinstance(csv_row, dict)

    # Restore from CSV row
    restored_article = NewsArticle.from_csv_row(csv_row)

    # Check that all fields match
    assert restored_article.id == original_article.id
    assert restored_article.title == original_article.title
    assert restored_article.content == original_article.content
    assert restored_article.url == original_article.url
    assert restored_article.published_at == original_article.published_at
    assert restored_article.source == original_article.source
    assert restored_article.category == original_article.category
    assert restored_article.raw_metadata == original_article.raw_metadata


def test_sentiment_analysis_creation():
    """Test SentimentAnalysis model creation."""
    sentiment = SentimentAnalysis(
        article_id="test-123",
        sentiment_score=0.5,
        confidence=0.8,
        key_phrases=["positive", "growth"],
        market_tone="bullish",
    )

    assert sentiment.article_id == "test-123"
    assert sentiment.sentiment_score == 0.5
    assert sentiment.market_tone == "bullish"


def test_sentiment_analysis_validation():
    """Test SentimentAnalysis validation."""
    # Valid sentiment should pass validation
    sentiment = SentimentAnalysis(
        article_id="test-123",
        sentiment_score=0.5,
        confidence=0.8,
        key_phrases=["positive", "growth"],
        market_tone="bullish",
    )
    assert sentiment.validate() is True

    # Invalid sentiment score should raise ValidationError
    with pytest.raises(ValidationError):
        invalid_sentiment = SentimentAnalysis(
            article_id="test-123",
            sentiment_score=2.0,  # Out of range
            confidence=0.8,
            key_phrases=["positive", "growth"],
            market_tone="bullish",
        )
        invalid_sentiment.validate()


def test_sentiment_analysis_json_serialization():
    """Test SentimentAnalysis JSON serialization and deserialization."""
    original_sentiment = SentimentAnalysis(
        article_id="test-123",
        sentiment_score=0.5,
        confidence=0.8,
        key_phrases=["positive", "growth"],
        market_tone="bullish",
    )

    # Serialize to JSON
    json_str = original_sentiment.to_json()
    assert isinstance(json_str, str)

    # Deserialize from JSON
    restored_sentiment = SentimentAnalysis.from_json(json_str)

    # Check that all fields match
    assert restored_sentiment.article_id == original_sentiment.article_id
    assert restored_sentiment.sentiment_score == original_sentiment.sentiment_score
    assert restored_sentiment.confidence == original_sentiment.confidence
    assert restored_sentiment.key_phrases == original_sentiment.key_phrases
    assert restored_sentiment.market_tone == original_sentiment.market_tone


def test_extracted_entity_creation():
    """Test ExtractedEntity model creation."""
    entity = ExtractedEntity(
        article_id="test-123",
        entity_type="stock_symbol",
        entity_value="AAPL",
        relevance_score=0.9,
        context="Apple stock mentioned",
    )

    assert entity.entity_type == "stock_symbol"
    assert entity.entity_value == "AAPL"
    assert entity.relevance_score == 0.9


def test_extracted_entity_validation():
    """Test ExtractedEntity validation."""
    # Valid entity should pass validation
    entity = ExtractedEntity(
        article_id="test-123",
        entity_type="stock_symbol",
        entity_value="AAPL",
        relevance_score=0.9,
        context="Apple stock mentioned",
    )
    assert entity.validate() is True

    # Invalid entity type should raise ValidationError
    with pytest.raises(ValidationError):
        invalid_entity = ExtractedEntity(
            article_id="test-123",
            entity_type="invalid_type",  # Invalid type
            entity_value="AAPL",
            relevance_score=0.9,
            context="Apple stock mentioned",
        )
        invalid_entity.validate()


def test_market_prediction_creation():
    """Test MarketPrediction model creation."""
    prediction = MarketPrediction(
        article_id="test-123",
        stock_symbol="AAPL",
        impact_direction="positive",
        impact_magnitude=0.7,
        confidence_level=0.8,
        reasoning="Strong earnings report",
        created_at=datetime.now(),
    )

    assert prediction.stock_symbol == "AAPL"
    assert prediction.impact_direction == "positive"
    assert prediction.confidence_level == 0.8


def test_market_prediction_validation():
    """Test MarketPrediction validation."""
    # Valid prediction should pass validation
    prediction = MarketPrediction(
        article_id="test-123",
        stock_symbol="AAPL",
        impact_direction="positive",
        impact_magnitude=0.7,
        confidence_level=0.8,
        reasoning="Strong earnings report",
        created_at=datetime.now(),
    )
    assert prediction.validate() is True

    # Invalid impact direction should raise ValidationError
    with pytest.raises(ValidationError):
        invalid_prediction = MarketPrediction(
            article_id="test-123",
            stock_symbol="AAPL",
            impact_direction="invalid_direction",  # Invalid direction
            impact_magnitude=0.7,
            confidence_level=0.8,
            reasoning="Strong earnings report",
            created_at=datetime.now(),
        )
        invalid_prediction.validate()


def test_market_prediction_json_serialization():
    """Test MarketPrediction JSON serialization and deserialization."""
    original_prediction = MarketPrediction(
        article_id="test-123",
        stock_symbol="AAPL",
        impact_direction="positive",
        impact_magnitude=0.7,
        confidence_level=0.8,
        reasoning="Strong earnings report",
        created_at=datetime(2023, 1, 1, 12, 0, 0),
    )

    # Serialize to JSON
    json_str = original_prediction.to_json()
    assert isinstance(json_str, str)

    # Deserialize from JSON
    restored_prediction = MarketPrediction.from_json(json_str)

    # Check that all fields match
    assert restored_prediction.article_id == original_prediction.article_id
    assert restored_prediction.stock_symbol == original_prediction.stock_symbol
    assert restored_prediction.impact_direction == original_prediction.impact_direction
    assert restored_prediction.impact_magnitude == original_prediction.impact_magnitude
    assert restored_prediction.confidence_level == original_prediction.confidence_level
    assert restored_prediction.reasoning == original_prediction.reasoning
    assert restored_prediction.created_at == original_prediction.created_at


def test_export_to_json():
    """Test export_to_json function."""
    articles = [
        NewsArticle(
            id="test-1",
            title="Test Article 1",
            content="Content 1",
            url="https://example.com/1",
            published_at=datetime(2023, 1, 1, 12, 0, 0),
            source="Yahoo Finance",
            category="Technology",
            raw_metadata={"test": "data1"},
        ),
        NewsArticle(
            id="test-2",
            title="Test Article 2",
            content="Content 2",
            url="https://example.com/2",
            published_at=datetime(2023, 1, 2, 12, 0, 0),
            source="Yahoo Finance",
            category="Finance",
            raw_metadata={"test": "data2"},
        ),
    ]

    json_str = export_to_json(articles)
    assert isinstance(json_str, str)

    # Parse JSON to verify it's valid
    data = json.loads(json_str)
    assert len(data) == 2
    assert data[0]["id"] == "test-1"
    assert data[1]["id"] == "test-2"


def test_export_to_csv():
    """Test export_to_csv function."""
    predictions = [
        MarketPrediction(
            article_id="test-1",
            stock_symbol="AAPL",
            impact_direction="positive",
            impact_magnitude=0.7,
            confidence_level=0.8,
            reasoning="Strong earnings",
            created_at=datetime(2023, 1, 1, 12, 0, 0),
        ),
        MarketPrediction(
            article_id="test-2",
            stock_symbol="GOOGL",
            impact_direction="negative",
            impact_magnitude=0.5,
            confidence_level=0.6,
            reasoning="Regulatory concerns",
            created_at=datetime(2023, 1, 2, 12, 0, 0),
        ),
    ]

    csv_str = export_to_csv(predictions)
    assert isinstance(csv_str, str)
    assert "article_id,stock_symbol" in csv_str  # Check header
    assert "test-1,AAPL" in csv_str  # Check data
    assert "test-2,GOOGL" in csv_str  # Check data
