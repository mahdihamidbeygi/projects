"""
Property-based tests for invalid input handling.

**Feature: news-market-predictor, Property 20: Invalid input handling**
"""

from datetime import datetime
from hypothesis import given, strategies as st, settings, assume

from news_market_predictor.predictor.market_predictor import BasicMarketPredictor
from news_market_predictor.models import (
    NewsArticle,
    SentimentAnalysis,
    ExtractedEntity,
    MarketPrediction,
)


# Strategy for generating invalid NewsArticle objects
@st.composite
def invalid_article_strategy(draw):
    """Generate invalid NewsArticle objects for testing."""
    invalid_type = draw(
        st.sampled_from(
            [
                "none",
                "missing_title",
                "missing_content",
                "missing_published_at",
                "empty_content",
                "short_content",
            ]
        )
    )

    if invalid_type == "none":
        return None

    # Base valid article
    base_article = NewsArticle(
        id=draw(st.text(min_size=1, max_size=50)),
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
        source="Yahoo Finance",
        category="markets",
        raw_metadata={},
    )

    # Make it invalid based on type
    if invalid_type == "missing_title":
        base_article.title = ""
    elif invalid_type == "missing_content":
        base_article.content = None
    elif invalid_type == "missing_published_at":
        base_article.published_at = None
    elif invalid_type == "empty_content":
        base_article.content = ""
    elif invalid_type == "short_content":
        base_article.content = "short"  # Less than 10 characters

    return base_article


# Strategy for generating invalid SentimentAnalysis objects
@st.composite
def invalid_sentiment_strategy(draw, article_id=None):
    """Generate invalid SentimentAnalysis objects for testing."""
    if article_id is None:
        article_id = draw(st.text(min_size=1, max_size=50))

    invalid_type = draw(
        st.sampled_from(
            [
                "none",
                "out_of_bounds_score_high",
                "out_of_bounds_score_low",
                "out_of_bounds_confidence_high",
                "out_of_bounds_confidence_low",
                "missing_score",
                "missing_confidence",
            ]
        )
    )

    if invalid_type == "none":
        return None

    # Base sentiment
    sentiment = SentimentAnalysis(
        article_id=article_id,
        sentiment_score=0.5,
        confidence=0.7,
        key_phrases=["test"],
        market_tone="neutral",
    )

    # Make it invalid
    if invalid_type == "out_of_bounds_score_high":
        sentiment.sentiment_score = draw(st.floats(min_value=1.1, max_value=10.0))
    elif invalid_type == "out_of_bounds_score_low":
        sentiment.sentiment_score = draw(st.floats(min_value=-10.0, max_value=-1.1))
    elif invalid_type == "out_of_bounds_confidence_high":
        sentiment.confidence = draw(st.floats(min_value=1.1, max_value=10.0))
    elif invalid_type == "out_of_bounds_confidence_low":
        sentiment.confidence = draw(st.floats(min_value=-10.0, max_value=-0.1))
    elif invalid_type == "missing_score":
        sentiment.sentiment_score = None
    elif invalid_type == "missing_confidence":
        sentiment.confidence = None

    return sentiment


# Strategy for generating invalid entity lists
@st.composite
def invalid_entities_strategy(draw, article_id=None):
    """Generate invalid entity lists for testing."""
    if article_id is None:
        article_id = draw(st.text(min_size=1, max_size=50))

    invalid_type = draw(
        st.sampled_from(
            [
                "none",
                "not_list",
                "missing_entity_type",
                "missing_entity_value",
                "out_of_bounds_relevance",
            ]
        )
    )

    if invalid_type == "none":
        return None

    if invalid_type == "not_list":
        return "not a list"

    # Create a stock entity
    stock_entity = ExtractedEntity(
        article_id=article_id,
        entity_type="stock_symbol",
        entity_value="AAPL",
        relevance_score=0.8,
        context="mentioned in article",
    )

    # Make it invalid
    if invalid_type == "missing_entity_type":
        stock_entity.entity_type = None
    elif invalid_type == "missing_entity_value":
        stock_entity.entity_value = None
    elif invalid_type == "out_of_bounds_relevance":
        stock_entity.relevance_score = draw(st.floats(min_value=1.1, max_value=10.0))

    return [stock_entity]


# Strategy for generating valid inputs for comparison
@st.composite
def valid_article_strategy(draw):
    """Generate valid NewsArticle objects for testing."""
    return NewsArticle(
        id=draw(st.text(min_size=1, max_size=50)),
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
        source="Yahoo Finance",
        category="markets",
        raw_metadata={},
    )


@st.composite
def valid_sentiment_strategy(draw, article_id=None):
    """Generate valid SentimentAnalysis objects for testing."""
    if article_id is None:
        article_id = draw(st.text(min_size=1, max_size=50))

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


@st.composite
def valid_entities_strategy(draw, article_id=None):
    """Generate valid entity lists with stock symbols for testing."""
    if article_id is None:
        article_id = draw(st.text(min_size=1, max_size=50))

    stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    stock_entity = ExtractedEntity(
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

    return [stock_entity]


@given(
    invalid_article=invalid_article_strategy(),
    valid_sentiment=valid_sentiment_strategy(),
    valid_entities=valid_entities_strategy(),
)
@settings(max_examples=100)
def test_invalid_article_returns_neutral_prediction(
    invalid_article, valid_sentiment, valid_entities
):
    """
    **Feature: news-market-predictor, Property 20: Invalid input handling**

    Property: For any invalid article input to prediction models, the system
    should return neutral predictions with appropriate error flags.

    **Validates: Requirements 5.3**

    This test verifies that when an invalid article is provided, the predictor
    handles it gracefully by returning empty list or neutral predictions.
    """
    predictor = BasicMarketPredictor()

    # Ensure matching IDs if article exists
    if invalid_article and hasattr(invalid_article, "id"):
        valid_sentiment.article_id = invalid_article.id
        for entity in valid_entities:
            entity.article_id = invalid_article.id

    # Generate predictions with invalid article
    predictions = predictor.predict_impact(
        invalid_article, valid_sentiment, valid_entities
    )

    # Should return a list (not crash)
    assert isinstance(
        predictions, list
    ), "Should return a list even with invalid article"

    # Should return empty list for invalid article
    assert len(predictions) == 0, "Should return empty list for invalid article input"


@given(
    valid_article=valid_article_strategy(),
    invalid_sentiment=invalid_sentiment_strategy(),
    valid_entities=valid_entities_strategy(),
)
@settings(max_examples=100)
def test_invalid_sentiment_returns_neutral_prediction(
    valid_article, invalid_sentiment, valid_entities
):
    """
    **Feature: news-market-predictor, Property 20: Invalid input handling**

    Property: For any invalid sentiment input to prediction models, the system
    should return neutral predictions with appropriate error flags.

    **Validates: Requirements 5.3**

    This test verifies that when invalid sentiment data is provided, the predictor
    returns neutral predictions with error information in the reasoning field.
    """
    predictor = BasicMarketPredictor()

    # Ensure matching IDs
    if invalid_sentiment and hasattr(invalid_sentiment, "article_id"):
        invalid_sentiment.article_id = valid_article.id
    for entity in valid_entities:
        entity.article_id = valid_article.id

    # Generate predictions with invalid sentiment
    predictions = predictor.predict_impact(
        valid_article, invalid_sentiment, valid_entities
    )

    # Should return a list (not crash)
    assert isinstance(
        predictions, list
    ), "Should return a list even with invalid sentiment"

    # If predictions are returned, they should be neutral with error flags
    for prediction in predictions:
        assert isinstance(prediction, MarketPrediction)
        assert (
            prediction.impact_direction == "neutral"
        ), "Invalid sentiment should result in neutral predictions"
        assert (
            prediction.confidence_level == 0.0
        ), "Invalid sentiment should result in zero confidence"
        assert (
            "Invalid" in prediction.reasoning or "invalid" in prediction.reasoning
        ), "Reasoning should indicate invalid input"


@given(
    valid_article=valid_article_strategy(),
    valid_sentiment=valid_sentiment_strategy(),
    invalid_entities=invalid_entities_strategy(),
)
@settings(max_examples=100)
def test_invalid_entities_returns_empty_or_neutral(
    valid_article, valid_sentiment, invalid_entities
):
    """
    **Feature: news-market-predictor, Property 20: Invalid input handling**

    Property: For any invalid entities input to prediction models, the system
    should handle gracefully and return empty list or neutral predictions.

    **Validates: Requirements 5.3**

    This test verifies that when invalid entity data is provided, the predictor
    handles it gracefully without crashing.
    """
    predictor = BasicMarketPredictor()

    # Ensure matching IDs
    valid_sentiment.article_id = valid_article.id
    if isinstance(invalid_entities, list):
        for entity in invalid_entities:
            if hasattr(entity, "article_id"):
                entity.article_id = valid_article.id

    # Generate predictions with invalid entities
    predictions = predictor.predict_impact(
        valid_article, valid_sentiment, invalid_entities
    )

    # Should return a list (not crash)
    assert isinstance(
        predictions, list
    ), "Should return a list even with invalid entities"

    # Should return empty list for invalid entities
    assert len(predictions) == 0, "Should return empty list for invalid entities input"


@given(
    valid_article=valid_article_strategy(),
    valid_sentiment=valid_sentiment_strategy(),
    valid_entities=valid_entities_strategy(),
)
@settings(max_examples=100)
def test_valid_inputs_do_not_return_error_flags(
    valid_article, valid_sentiment, valid_entities
):
    """
    **Feature: news-market-predictor, Property 20: Invalid input handling**

    Property: For any valid inputs, the system should NOT return predictions
    with error flags, ensuring error flags are only used for invalid inputs.

    **Validates: Requirements 5.3**

    This test verifies that valid inputs produce normal predictions without
    error indicators, confirming that error handling is specific to invalid inputs.
    """
    predictor = BasicMarketPredictor()

    # Ensure matching IDs
    valid_sentiment.article_id = valid_article.id
    for entity in valid_entities:
        entity.article_id = valid_article.id

    # Generate predictions with valid inputs
    predictions = predictor.predict_impact(
        valid_article, valid_sentiment, valid_entities
    )

    # Should return predictions
    assert isinstance(predictions, list), "Should return a list"
    assert len(predictions) > 0, "Should return predictions for valid inputs"

    # Predictions should not have error flags
    for prediction in predictions:
        assert isinstance(prediction, MarketPrediction)
        # Should not have "Invalid" or "Error" in reasoning for valid inputs
        reasoning_lower = prediction.reasoning.lower()
        assert (
            "invalid input" not in reasoning_lower
        ), "Valid inputs should not produce 'invalid input' error flags"
        assert (
            "error in prediction generation" not in reasoning_lower
        ), "Valid inputs should not produce 'error in prediction generation' flags"


@given(
    invalid_article=invalid_article_strategy(),
    invalid_sentiment=invalid_sentiment_strategy(),
    invalid_entities=invalid_entities_strategy(),
)
@settings(max_examples=50)
def test_multiple_invalid_inputs_handled_gracefully(
    invalid_article, invalid_sentiment, invalid_entities
):
    """
    **Feature: news-market-predictor, Property 20: Invalid input handling**

    Property: For any combination of invalid inputs, the system should handle
    them gracefully without crashing and return appropriate responses.

    **Validates: Requirements 5.3**

    This test verifies that even when multiple inputs are invalid simultaneously,
    the predictor handles the situation gracefully.
    """
    predictor = BasicMarketPredictor()

    # Ensure matching IDs if objects exist
    if invalid_article and hasattr(invalid_article, "id"):
        if invalid_sentiment and hasattr(invalid_sentiment, "article_id"):
            invalid_sentiment.article_id = invalid_article.id
        if isinstance(invalid_entities, list):
            for entity in invalid_entities:
                if hasattr(entity, "article_id"):
                    entity.article_id = invalid_article.id

    # Generate predictions with multiple invalid inputs
    # Should not crash
    try:
        predictions = predictor.predict_impact(
            invalid_article, invalid_sentiment, invalid_entities
        )

        # Should return a list
        assert isinstance(
            predictions, list
        ), "Should return a list even with multiple invalid inputs"

        # All predictions should be neutral or list should be empty
        for prediction in predictions:
            assert isinstance(prediction, MarketPrediction)
            assert (
                prediction.impact_direction == "neutral"
            ), "Multiple invalid inputs should result in neutral predictions"
            assert (
                prediction.confidence_level == 0.0
            ), "Multiple invalid inputs should result in zero confidence"

    except Exception as e:
        # Should not raise exceptions for invalid inputs
        assert False, f"Should not raise exception for invalid inputs: {e}"


@given(
    valid_article=valid_article_strategy(),
    valid_sentiment=valid_sentiment_strategy(),
    valid_entities=valid_entities_strategy(),
)
@settings(max_examples=100)
def test_error_flags_contain_descriptive_information(
    valid_article, valid_sentiment, valid_entities
):
    """
    **Feature: news-market-predictor, Property 20: Invalid input handling**

    Property: For any invalid input that generates a neutral prediction with
    error flags, the error information should be descriptive and helpful.

    **Validates: Requirements 5.3**

    This test verifies that error flags provide meaningful information about
    what went wrong with the input.
    """
    predictor = BasicMarketPredictor()

    # Create an invalid sentiment by setting out-of-bounds score
    invalid_sentiment = SentimentAnalysis(
        article_id=valid_article.id,
        sentiment_score=5.0,  # Out of bounds
        confidence=0.7,
        key_phrases=["test"],
        market_tone="neutral",
    )

    for entity in valid_entities:
        entity.article_id = valid_article.id

    # Generate predictions with invalid sentiment
    predictions = predictor.predict_impact(
        valid_article, invalid_sentiment, valid_entities
    )

    # Check error flags in predictions
    for prediction in predictions:
        if (
            prediction.impact_direction == "neutral"
            and prediction.confidence_level == 0.0
        ):
            # Should have descriptive reasoning
            assert (
                len(prediction.reasoning) > 0
            ), "Error predictions should have non-empty reasoning"
            assert isinstance(prediction.reasoning, str), "Reasoning should be a string"
            # Should contain some indication of the problem
            reasoning_lower = prediction.reasoning.lower()
            assert any(
                keyword in reasoning_lower
                for keyword in ["invalid", "error", "failed", "problem"]
            ), "Error reasoning should contain descriptive keywords"


@given(st.data())
@settings(max_examples=50)
def test_invalid_input_handling_is_consistent(data):
    """
    **Feature: news-market-predictor, Property 20: Invalid input handling**

    Property: For any invalid input, the handling behavior should be consistent
    across multiple calls with the same invalid input.

    **Validates: Requirements 5.3**

    This test verifies that invalid input handling is deterministic and consistent.
    """
    predictor = BasicMarketPredictor()

    # Generate an invalid article
    invalid_article = data.draw(invalid_article_strategy())
    valid_sentiment = data.draw(valid_sentiment_strategy())
    valid_entities = data.draw(valid_entities_strategy())

    # Ensure matching IDs
    if invalid_article and hasattr(invalid_article, "id"):
        valid_sentiment.article_id = invalid_article.id
        for entity in valid_entities:
            entity.article_id = invalid_article.id

    # Call predict_impact multiple times with same invalid input
    predictions1 = predictor.predict_impact(
        invalid_article, valid_sentiment, valid_entities
    )
    predictions2 = predictor.predict_impact(
        invalid_article, valid_sentiment, valid_entities
    )
    predictions3 = predictor.predict_impact(
        invalid_article, valid_sentiment, valid_entities
    )

    # All calls should return the same result
    assert (
        len(predictions1) == len(predictions2) == len(predictions3)
    ), "Invalid input handling should be consistent across calls"

    # If predictions are returned, they should be identical
    for p1, p2, p3 in zip(predictions1, predictions2, predictions3):
        assert p1.impact_direction == p2.impact_direction == p3.impact_direction
        assert p1.confidence_level == p2.confidence_level == p3.confidence_level
        assert p1.stock_symbol == p2.stock_symbol == p3.stock_symbol
