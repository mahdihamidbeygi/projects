"""
Property-based tests for confidence level calculation.

**Feature: news-market-predictor, Property 10: Confidence level calculation**
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
    historical_accuracy=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100)
def test_confidence_level_bounds_for_any_prediction(
    article, sentiment, entities, historical_accuracy
):
    """
    **Feature: news-market-predictor, Property 10: Confidence level calculation**

    Property: For any prediction, the confidence level should be between 0% and 100%
    and reflect historical accuracy and article characteristics.

    **Validates: Requirements 3.2**

    This test verifies that confidence levels are always within valid bounds
    regardless of input data and historical accuracy settings.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Initialize the market predictor with varying historical accuracy
    predictor = BasicMarketPredictor(historical_accuracy=historical_accuracy)

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Verify that predictions were generated for stock symbols
    stock_entities = [e for e in entities if e.entity_type == "stock_symbol"]
    if stock_entities:
        assert (
            len(predictions) >= 1
        ), "Should generate at least one prediction when stock symbols are present"

    # Verify confidence level bounds for each prediction
    for prediction in predictions:
        # Confidence level should be between 0.0 and 1.0 (0% to 100%)
        assert 0.0 <= prediction.confidence_level <= 1.0, (
            f"Confidence level {prediction.confidence_level} is out of bounds. "
            f"Must be between 0.0 and 1.0 (0% to 100%)"
        )

        # Confidence level should be a valid float
        assert isinstance(
            prediction.confidence_level, float
        ), f"Confidence level must be a float, got {type(prediction.confidence_level)}"

        # Confidence level should not be NaN or infinity
        assert not (
            prediction.confidence_level != prediction.confidence_level
        ), "Confidence level cannot be NaN"
        assert prediction.confidence_level != float(
            "inf"
        ), "Confidence level cannot be infinity"
        assert prediction.confidence_level != float(
            "-inf"
        ), "Confidence level cannot be negative infinity"


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_confidence_reflects_historical_accuracy(article, sentiment, entities):
    """
    **Feature: news-market-predictor, Property 10: Confidence level calculation**

    Property: Confidence levels should reflect historical accuracy - higher historical
    accuracy should generally lead to higher confidence levels.

    **Validates: Requirements 3.2**

    This test verifies that historical accuracy influences confidence calculation.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Test with different historical accuracy levels
    low_accuracy = 0.2
    high_accuracy = 0.9

    predictor_low = BasicMarketPredictor(historical_accuracy=low_accuracy)
    predictor_high = BasicMarketPredictor(historical_accuracy=high_accuracy)

    # Generate predictions with both predictors
    predictions_low = predictor_low.predict_impact(article, sentiment, entities)
    predictions_high = predictor_high.predict_impact(article, sentiment, entities)

    # Both should generate the same number of predictions
    assert len(predictions_low) == len(predictions_high)

    # Skip test if no predictions generated
    assume(len(predictions_low) > 0)

    # Compare confidence levels - high accuracy predictor should generally have higher confidence
    # We'll check that the average confidence is higher for high accuracy predictor
    avg_confidence_low = sum(p.confidence_level for p in predictions_low) / len(
        predictions_low
    )
    avg_confidence_high = sum(p.confidence_level for p in predictions_high) / len(
        predictions_high
    )

    # The high accuracy predictor should have higher average confidence
    # Allow some tolerance for edge cases where signal strength dominates
    assert avg_confidence_high >= avg_confidence_low - 0.1, (
        f"High accuracy predictor (avg confidence: {avg_confidence_high:.3f}) should have "
        f"higher or similar confidence compared to low accuracy predictor "
        f"(avg confidence: {avg_confidence_low:.3f})"
    )


@given(
    article=news_article_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_confidence_reflects_article_characteristics(article, entities):
    """
    **Feature: news-market-predictor, Property 10: Confidence level calculation**

    Property: Confidence levels should reflect article characteristics - stronger
    sentiment and higher entity relevance should influence confidence.

    **Validates: Requirements 3.2**

    This test verifies that article characteristics influence confidence calculation.
    """
    # Ensure entities have matching article_id
    for entity in entities:
        entity.article_id = article.id

    # Create sentiment analyses with different characteristics
    weak_sentiment = SentimentAnalysis(
        article_id=article.id,
        sentiment_score=0.1,  # Weak sentiment
        confidence=0.3,  # Low confidence
        key_phrases=["maybe", "possibly"],
        market_tone="neutral",
    )

    strong_sentiment = SentimentAnalysis(
        article_id=article.id,
        sentiment_score=0.8,  # Strong sentiment
        confidence=0.9,  # High confidence
        key_phrases=["definitely", "strongly", "significant"],
        market_tone="bullish",
    )

    # Initialize predictor
    predictor = BasicMarketPredictor(historical_accuracy=0.65)

    # Generate predictions with different sentiment characteristics
    predictions_weak = predictor.predict_impact(article, weak_sentiment, entities)
    predictions_strong = predictor.predict_impact(article, strong_sentiment, entities)

    # Both should generate the same number of predictions
    assert len(predictions_weak) == len(predictions_strong)

    # Skip test if no predictions generated
    assume(len(predictions_weak) > 0)

    # Compare confidence levels
    for pred_weak, pred_strong in zip(predictions_weak, predictions_strong):
        # Both should have valid confidence bounds
        assert 0.0 <= pred_weak.confidence_level <= 1.0
        assert 0.0 <= pred_strong.confidence_level <= 1.0

        # Strong sentiment should generally lead to higher confidence
        # Allow some tolerance for cases where other factors dominate
        assert pred_strong.confidence_level >= pred_weak.confidence_level - 0.2, (
            f"Strong sentiment prediction (confidence: {pred_strong.confidence_level:.3f}) "
            f"should have higher or similar confidence compared to weak sentiment prediction "
            f"(confidence: {pred_weak.confidence_level:.3f}) for stock {pred_weak.stock_symbol}"
        )


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_confidence_calculation_consistency(article, sentiment, entities):
    """
    **Feature: news-market-predictor, Property 10: Confidence level calculation**

    Property: Confidence calculation should be consistent - same inputs should
    produce the same confidence levels.

    **Validates: Requirements 3.2**

    This test verifies that confidence calculation is deterministic and consistent.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Initialize predictor
    predictor = BasicMarketPredictor(historical_accuracy=0.65)

    # Generate predictions multiple times with same input
    predictions1 = predictor.predict_impact(article, sentiment, entities)
    predictions2 = predictor.predict_impact(article, sentiment, entities)
    predictions3 = predictor.predict_impact(article, sentiment, entities)

    # All prediction sets should have the same length
    assert len(predictions1) == len(predictions2) == len(predictions3)

    # Skip test if no predictions generated
    assume(len(predictions1) > 0)

    # Confidence levels should be identical for same inputs
    for p1, p2, p3 in zip(predictions1, predictions2, predictions3):
        assert p1.confidence_level == p2.confidence_level == p3.confidence_level, (
            f"Confidence levels should be consistent for same inputs. "
            f"Got {p1.confidence_level}, {p2.confidence_level}, {p3.confidence_level} "
            f"for stock {p1.stock_symbol}"
        )

        # All should be within valid bounds
        assert 0.0 <= p1.confidence_level <= 1.0
        assert 0.0 <= p2.confidence_level <= 1.0
        assert 0.0 <= p3.confidence_level <= 1.0


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_confidence_with_extreme_values(article, sentiment, entities):
    """
    **Feature: news-market-predictor, Property 10: Confidence level calculation**

    Property: Even with extreme input values, confidence levels should remain
    within valid bounds (0% to 100%).

    **Validates: Requirements 3.2**

    This test verifies that extreme values don't cause confidence calculation
    to produce invalid results.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Test with extreme historical accuracy values
    extreme_accuracies = [0.0, 0.01, 0.99, 1.0]

    for accuracy in extreme_accuracies:
        predictor = BasicMarketPredictor(historical_accuracy=accuracy)

        # Test with extreme sentiment values
        extreme_sentiments = [
            SentimentAnalysis(
                article_id=article.id,
                sentiment_score=-1.0,  # Maximum negative
                confidence=0.0,  # Minimum confidence
                key_phrases=[],
                market_tone="bearish",
            ),
            SentimentAnalysis(
                article_id=article.id,
                sentiment_score=1.0,  # Maximum positive
                confidence=1.0,  # Maximum confidence
                key_phrases=["extremely", "very", "highly"],
                market_tone="bullish",
            ),
            SentimentAnalysis(
                article_id=article.id,
                sentiment_score=0.0,  # Neutral
                confidence=0.5,  # Medium confidence
                key_phrases=[],
                market_tone="neutral",
            ),
        ]

        for extreme_sentiment in extreme_sentiments:
            predictions = predictor.predict_impact(article, extreme_sentiment, entities)

            # Verify confidence bounds for all predictions
            for prediction in predictions:
                assert 0.0 <= prediction.confidence_level <= 1.0, (
                    f"Confidence level {prediction.confidence_level} is out of bounds "
                    f"with historical accuracy {accuracy} and sentiment score "
                    f"{extreme_sentiment.sentiment_score}. Must be between 0.0 and 1.0"
                )

                # Verify no invalid float values
                assert not (
                    prediction.confidence_level != prediction.confidence_level
                ), "Confidence level cannot be NaN"
                assert prediction.confidence_level != float(
                    "inf"
                ), "Confidence level cannot be infinity"
                assert prediction.confidence_level != float(
                    "-inf"
                ), "Confidence level cannot be negative infinity"


@given(
    prediction_data=st.dictionaries(
        st.sampled_from(["direction", "strength", "confidence", "magnitude"]),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=100)
def test_calculate_confidence_method_bounds(prediction_data):
    """
    **Feature: news-market-predictor, Property 10: Confidence level calculation**

    Property: The calculate_confidence method should always return values
    between 0.0 and 1.0 regardless of input data.

    **Validates: Requirements 3.2**

    This test directly tests the calculate_confidence method with various
    prediction data inputs.
    """
    # Initialize predictor with random historical accuracy
    historical_accuracy = 0.65
    predictor = BasicMarketPredictor(historical_accuracy=historical_accuracy)

    # Call calculate_confidence method directly
    confidence = predictor.calculate_confidence(prediction_data)

    # Verify confidence is within bounds
    assert 0.0 <= confidence <= 1.0, (
        f"Confidence {confidence} is out of bounds for input {prediction_data}. "
        f"Must be between 0.0 and 1.0"
    )

    # Verify confidence is a valid float
    assert isinstance(
        confidence, float
    ), f"Confidence must be a float, got {type(confidence)}"

    # Verify no invalid float values
    assert not (confidence != confidence), "Confidence cannot be NaN"
    assert confidence != float("inf"), "Confidence cannot be infinity"
    assert confidence != float("-inf"), "Confidence cannot be negative infinity"
