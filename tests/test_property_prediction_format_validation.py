"""
Property-based tests for prediction format validation.

**Feature: news-market-predictor, Property 9: Prediction format validation**
"""

from datetime import datetime
from hypothesis import given, strategies as st, settings

from news_market_predictor.predictor.market_predictor import BasicMarketPredictor
from news_market_predictor.models import (
    NewsArticle,
    SentimentAnalysis,
    ExtractedEntity,
    MarketPrediction,
)


# Strategy for generating valid NewsArticle objects
@st.composite
def news_article_strategy(draw):
    """Generate valid NewsArticle objects for testing."""
    return NewsArticle(
        id=draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            )
        ),
        title=draw(st.text(min_size=1, max_size=200)),
        content=draw(st.text(min_size=10, max_size=2000)),
        url=draw(
            st.text(min_size=10, max_size=100).map(
                lambda x: f"https://finance.yahoo.com/{x.replace(' ', '-')}"
            )
        ),
        published_at=draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
            )
        ),
        source=draw(
            st.sampled_from(["Yahoo Finance", "Reuters", "Bloomberg", "MarketWatch"])
        ),
        category=draw(
            st.sampled_from(
                ["earnings", "markets", "technology", "healthcare", "finance"]
            )
        ),
        raw_metadata=draw(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.text(min_size=1, max_size=50),
                min_size=0,
                max_size=5,
            )
        ),
    )


# Strategy for generating valid SentimentAnalysis objects
@st.composite
def sentiment_analysis_strategy(draw, article_id=None):
    """Generate valid SentimentAnalysis objects for testing."""
    if article_id is None:
        article_id = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            )
        )

    return SentimentAnalysis(
        article_id=article_id,
        sentiment_score=draw(
            st.floats(
                min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        confidence=draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        key_phrases=draw(
            st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=10)
        ),
        market_tone=draw(st.sampled_from(["bullish", "bearish", "neutral"])),
    )


# Strategy for generating valid ExtractedEntity objects with stock symbols
@st.composite
def stock_entity_strategy(draw, article_id=None):
    """Generate valid ExtractedEntity objects with stock_symbol type for testing."""
    if article_id is None:
        article_id = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            )
        )

    stock_symbols = [
        "AAPL",
        "GOOGL",
        "MSFT",
        "TSLA",
        "AMZN",
        "META",
        "NVDA",
        "NFLX",
        "AMD",
        "INTC",
    ]

    return ExtractedEntity(
        article_id=article_id,
        entity_type="stock_symbol",
        entity_value=draw(st.sampled_from(stock_symbols)),
        relevance_score=draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        context=draw(st.text(min_size=1, max_size=200)),
    )


# Strategy for generating lists of entities including at least one stock symbol
@st.composite
def entities_with_stock_strategy(draw, article_id=None):
    """Generate list of entities that includes at least one stock symbol."""
    if article_id is None:
        article_id = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            )
        )

    # Always include at least one stock symbol
    stock_entity = draw(stock_entity_strategy(article_id))
    entities = [stock_entity]

    # Optionally add more entities
    additional_entities = draw(
        st.lists(
            st.one_of(
                stock_entity_strategy(article_id),
                st.builds(
                    ExtractedEntity,
                    article_id=st.just(article_id),
                    entity_type=st.sampled_from(["company", "metric"]),
                    entity_value=st.text(min_size=1, max_size=50),
                    relevance_score=st.floats(
                        min_value=0.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    context=st.text(min_size=1, max_size=200),
                ),
            ),
            min_size=0,
            max_size=5,
        )
    )

    entities.extend(additional_entities)
    return entities


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_prediction_format_validation_for_any_input(article, sentiment, entities):
    """
    **Feature: news-market-predictor, Property 9: Prediction format validation**

    Property: For any generated prediction, the impact direction should be one of:
    positive, negative, or neutral.

    **Validates: Requirements 3.1**

    This test verifies that regardless of input data (article content, sentiment,
    entities), the market predictor always produces predictions with valid
    impact_direction values.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.entity_id = article.id

    # Initialize the market predictor
    predictor = BasicMarketPredictor()

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Verify that predictions were generated (should have at least one for stock symbols)
    stock_entities = [e for e in entities if e.entity_type == "stock_symbol"]
    if stock_entities:
        assert (
            len(predictions) >= 1
        ), "Should generate at least one prediction when stock symbols are present"

    # Verify each prediction has valid format
    valid_directions = {"positive", "negative", "neutral"}

    for prediction in predictions:
        # Verify prediction is a MarketPrediction object
        assert isinstance(prediction, MarketPrediction)

        # Verify impact_direction is one of the valid values
        assert prediction.impact_direction in valid_directions, (
            f"Impact direction '{prediction.impact_direction}' is not valid. "
            f"Must be one of: {valid_directions}"
        )

        # Verify impact_direction is a string
        assert isinstance(prediction.impact_direction, str)

        # Verify the prediction passes model validation
        assert prediction.validate() is True


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_prediction_format_validation_with_extreme_sentiment(
    article, sentiment, entities
):
    """
    **Feature: news-market-predictor, Property 9: Prediction format validation**

    Property: Even with extreme sentiment scores, predictions should maintain
    valid impact_direction format.

    **Validates: Requirements 3.1**

    This test verifies that extreme sentiment values don't cause invalid
    impact_direction values to be generated.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.entity_id = article.id

    # Force extreme sentiment values
    extreme_sentiments = [-1.0, -0.9, 0.0, 0.9, 1.0]

    for extreme_score in extreme_sentiments:
        # Create sentiment with extreme score
        extreme_sentiment = SentimentAnalysis(
            article_id=article.id,
            sentiment_score=extreme_score,
            confidence=sentiment.confidence,
            key_phrases=sentiment.key_phrases,
            market_tone=sentiment.market_tone,
        )

        # Initialize the market predictor
        predictor = BasicMarketPredictor()

        # Generate predictions
        predictions = predictor.predict_impact(article, extreme_sentiment, entities)

        # Verify format for each prediction
        valid_directions = {"positive", "negative", "neutral"}

        for prediction in predictions:
            assert prediction.impact_direction in valid_directions, (
                f"Impact direction '{prediction.impact_direction}' is not valid for "
                f"extreme sentiment {extreme_score}. Must be one of: {valid_directions}"
            )


@given(article=news_article_strategy(), sentiment=sentiment_analysis_strategy())
@settings(max_examples=100)
def test_prediction_format_validation_with_no_stock_entities(article, sentiment):
    """
    **Feature: news-market-predictor, Property 9: Prediction format validation**

    Property: When no stock entities are present, the predictor should return
    an empty list rather than predictions with invalid format.

    **Validates: Requirements 3.1**

    This test verifies proper handling when no stock symbols are found.
    """
    # Ensure sentiment has matching article_id
    sentiment.article_id = article.id

    # Create entities without stock symbols
    entities = [
        ExtractedEntity(
            article_id=article.id,
            entity_type="company",
            entity_value="Some Company Inc",
            relevance_score=0.8,
            context="mentioned in article",
        ),
        ExtractedEntity(
            article_id=article.id,
            entity_type="metric",
            entity_value="revenue",
            relevance_score=0.6,
            context="financial metric",
        ),
    ]

    # Initialize the market predictor
    predictor = BasicMarketPredictor()

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Should return empty list when no stock symbols present
    assert isinstance(predictions, list)
    assert (
        len(predictions) == 0
    ), "Should return empty list when no stock symbols are present"


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_prediction_format_validation_consistency(article, sentiment, entities):
    """
    **Feature: news-market-predictor, Property 9: Prediction format validation**

    Property: Multiple calls with the same input should produce predictions
    with consistent format validation.

    **Validates: Requirements 3.1**

    This test verifies that format validation is consistent across multiple
    prediction generations.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.entity_id = article.id

    # Initialize the market predictor
    predictor = BasicMarketPredictor()

    # Generate predictions multiple times
    predictions1 = predictor.predict_impact(article, sentiment, entities)
    predictions2 = predictor.predict_impact(article, sentiment, entities)
    predictions3 = predictor.predict_impact(article, sentiment, entities)

    # All prediction sets should have the same length
    assert len(predictions1) == len(predictions2) == len(predictions3)

    # All predictions should have valid format
    valid_directions = {"positive", "negative", "neutral"}

    for predictions in [predictions1, predictions2, predictions3]:
        for prediction in predictions:
            assert prediction.impact_direction in valid_directions
            assert isinstance(prediction.impact_direction, str)
            assert prediction.validate() is True

    # Predictions should be consistent (same input -> same output)
    for p1, p2, p3 in zip(predictions1, predictions2, predictions3):
        assert p1.impact_direction == p2.impact_direction == p3.impact_direction
        assert p1.stock_symbol == p2.stock_symbol == p3.stock_symbol


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_prediction_format_validation_with_error_conditions(
    article, sentiment, entities
):
    """
    **Feature: news-market-predictor, Property 9: Prediction format validation**

    Property: Even when errors occur during prediction generation, any returned
    predictions should still have valid format.

    **Validates: Requirements 3.1**

    This test verifies that error handling doesn't compromise format validation.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.entity_id = article.id

    # Initialize the market predictor
    predictor = BasicMarketPredictor()

    # Test with potentially problematic data
    # Empty content
    empty_article = NewsArticle(
        id=article.id,
        title=article.title,
        content="",  # Empty content
        url=article.url,
        published_at=article.published_at,
        source=article.source,
        category=article.category,
        raw_metadata=article.raw_metadata,
    )

    # Generate predictions with empty content
    predictions = predictor.predict_impact(empty_article, sentiment, entities)

    # Verify format even with empty content
    valid_directions = {"positive", "negative", "neutral"}

    for prediction in predictions:
        assert isinstance(prediction, MarketPrediction)
        assert prediction.impact_direction in valid_directions
        assert isinstance(prediction.impact_direction, str)
        assert prediction.validate() is True
