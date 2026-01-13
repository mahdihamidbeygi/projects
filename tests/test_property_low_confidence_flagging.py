"""
Property-based tests for low confidence flagging.

**Feature: news-market-predictor, Property 13: Low confidence flagging**
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


# Strategy for generating low confidence scenarios
@st.composite
def low_confidence_scenario_strategy(draw):
    """Generate scenarios that should result in low confidence predictions."""
    article = draw(news_article_strategy())

    # Create conditions that typically lead to low confidence:
    # 1. Low historical accuracy
    # 2. Weak sentiment with low confidence
    # 3. Low entity relevance scores

    historical_accuracy = draw(st.floats(min_value=0.0, max_value=0.4))  # Low accuracy

    sentiment = SentimentAnalysis(
        article_id=article.id,
        sentiment_score=draw(
            st.floats(min_value=-0.2, max_value=0.2)
        ),  # Weak sentiment
        confidence=draw(st.floats(min_value=0.0, max_value=0.3)),  # Low confidence
        key_phrases=draw(
            st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=2)
        ),
        market_tone="neutral",
    )

    # Create entities with low relevance scores
    entities = []
    stock_entity = ExtractedEntity(
        article_id=article.id,
        entity_type="stock_symbol",
        entity_value=draw(st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA"])),
        relevance_score=draw(st.floats(min_value=0.0, max_value=0.4)),  # Low relevance
        context=draw(st.text(min_size=1, max_size=100)),
    )
    entities.append(stock_entity)

    return article, sentiment, entities, historical_accuracy


# Strategy for generating high confidence scenarios
@st.composite
def high_confidence_scenario_strategy(draw):
    """Generate scenarios that should result in high confidence predictions."""
    article = draw(news_article_strategy())

    # Create conditions that typically lead to high confidence:
    # 1. High historical accuracy
    # 2. Strong sentiment with high confidence
    # 3. High entity relevance scores

    historical_accuracy = draw(st.floats(min_value=0.7, max_value=1.0))  # High accuracy

    sentiment = SentimentAnalysis(
        article_id=article.id,
        sentiment_score=draw(
            st.floats(min_value=0.5, max_value=1.0)
        ),  # Strong sentiment
        confidence=draw(st.floats(min_value=0.7, max_value=1.0)),  # High confidence
        key_phrases=draw(
            st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=5)
        ),
        market_tone=draw(st.sampled_from(["bullish", "bearish"])),
    )

    # Create entities with high relevance scores
    entities = []
    stock_entity = ExtractedEntity(
        article_id=article.id,
        entity_type="stock_symbol",
        entity_value=draw(st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA"])),
        relevance_score=draw(st.floats(min_value=0.7, max_value=1.0)),  # High relevance
        context=draw(st.text(min_size=1, max_size=100)),
    )
    entities.append(stock_entity)

    return article, sentiment, entities, historical_accuracy


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_low_confidence_flagging_for_any_prediction(article, sentiment, entities):
    """
    **Feature: news-market-predictor, Property 13: Low confidence flagging**

    Property: For any prediction with confidence below 30%, the system should flag it as low-confidence.

    **Validates: Requirements 3.5**

    This test verifies that predictions with confidence below the threshold (30%)
    are properly flagged as low-confidence in their reasoning text.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Initialize the market predictor
    predictor = BasicMarketPredictor(historical_accuracy=0.65)

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Skip test if no predictions generated
    assume(len(predictions) > 0)

    # Check each prediction for proper low confidence flagging
    for prediction in predictions:
        # Verify confidence level is valid
        assert (
            0.0 <= prediction.confidence_level <= 1.0
        ), f"Confidence level {prediction.confidence_level} is out of bounds"

        # Check low confidence flagging
        if prediction.confidence_level < 0.30:
            # Low confidence predictions should be flagged in reasoning
            assert "[LOW CONFIDENCE:" in prediction.reasoning, (
                f"Prediction with confidence {prediction.confidence_level:.3f} "
                f"should be flagged as low confidence in reasoning. "
                f"Reasoning: '{prediction.reasoning}'"
            )

            # The flag should include the confidence value
            confidence_str = f"{prediction.confidence_level:.2f}"
            assert confidence_str in prediction.reasoning, (
                f"Low confidence flag should include confidence value {confidence_str}. "
                f"Reasoning: '{prediction.reasoning}'"
            )
        else:
            # High confidence predictions should not be flagged
            assert "[LOW CONFIDENCE:" not in prediction.reasoning, (
                f"Prediction with confidence {prediction.confidence_level:.3f} "
                f"should not be flagged as low confidence. "
                f"Reasoning: '{prediction.reasoning}'"
            )


@given(scenario=low_confidence_scenario_strategy())
@settings(max_examples=100)
def test_low_confidence_scenarios_are_flagged(scenario):
    """
    **Feature: news-market-predictor, Property 13: Low confidence flagging**

    Property: Scenarios designed to produce low confidence should result in
    flagged predictions.

    **Validates: Requirements 3.5**

    This test uses scenarios specifically designed to produce low confidence
    predictions and verifies they are properly flagged.
    """
    article, sentiment, entities, historical_accuracy = scenario

    # Initialize predictor with low historical accuracy
    predictor = BasicMarketPredictor(historical_accuracy=historical_accuracy)

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Skip test if no predictions generated
    assume(len(predictions) > 0)

    # At least some predictions should have low confidence and be flagged
    low_confidence_predictions = [p for p in predictions if p.confidence_level < 0.30]

    # If we have low confidence predictions, they should be flagged
    for prediction in low_confidence_predictions:
        assert "[LOW CONFIDENCE:" in prediction.reasoning, (
            f"Low confidence prediction (confidence: {prediction.confidence_level:.3f}) "
            f"should be flagged. Reasoning: '{prediction.reasoning}'"
        )


@given(scenario=high_confidence_scenario_strategy())
@settings(max_examples=100)
def test_high_confidence_scenarios_are_not_flagged(scenario):
    """
    **Feature: news-market-predictor, Property 13: Low confidence flagging**

    Property: Scenarios designed to produce high confidence should not result in
    flagged predictions.

    **Validates: Requirements 3.5**

    This test uses scenarios specifically designed to produce high confidence
    predictions and verifies they are not flagged as low confidence.
    """
    article, sentiment, entities, historical_accuracy = scenario

    # Initialize predictor with high historical accuracy
    predictor = BasicMarketPredictor(historical_accuracy=historical_accuracy)

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Skip test if no predictions generated
    assume(len(predictions) > 0)

    # High confidence predictions should not be flagged
    high_confidence_predictions = [p for p in predictions if p.confidence_level >= 0.30]

    for prediction in high_confidence_predictions:
        assert "[LOW CONFIDENCE:" not in prediction.reasoning, (
            f"High confidence prediction (confidence: {prediction.confidence_level:.3f}) "
            f"should not be flagged as low confidence. "
            f"Reasoning: '{prediction.reasoning}'"
        )


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_threshold_boundary_behavior(article, sentiment, entities):
    """
    **Feature: news-market-predictor, Property 13: Low confidence flagging**

    Property: The flagging behavior should be consistent at the 30% threshold boundary.

    **Validates: Requirements 3.5**

    This test verifies that the 30% threshold is applied correctly and consistently.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Test with different historical accuracies to get predictions near the threshold
    threshold = 0.30

    for historical_accuracy in [0.1, 0.2, 0.3, 0.4, 0.5]:
        predictor = BasicMarketPredictor(historical_accuracy=historical_accuracy)
        predictions = predictor.predict_impact(article, sentiment, entities)

        for prediction in predictions:
            # Verify threshold behavior
            if prediction.confidence_level < threshold:
                assert "[LOW CONFIDENCE:" in prediction.reasoning, (
                    f"Prediction with confidence {prediction.confidence_level:.3f} "
                    f"(< {threshold}) should be flagged as low confidence"
                )
            elif prediction.confidence_level >= threshold:
                assert "[LOW CONFIDENCE:" not in prediction.reasoning, (
                    f"Prediction with confidence {prediction.confidence_level:.3f} "
                    f"(>= {threshold}) should not be flagged as low confidence"
                )


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_flag_format_consistency(article, sentiment, entities):
    """
    **Feature: news-market-predictor, Property 13: Low confidence flagging**

    Property: Low confidence flags should have consistent format across all predictions.

    **Validates: Requirements 3.5**

    This test verifies that the low confidence flag format is consistent and
    includes the confidence value.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Use low historical accuracy to increase chances of low confidence predictions
    predictor = BasicMarketPredictor(historical_accuracy=0.2)

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Skip test if no predictions generated
    assume(len(predictions) > 0)

    # Check format consistency for low confidence predictions
    low_confidence_predictions = [p for p in predictions if p.confidence_level < 0.30]

    for prediction in low_confidence_predictions:
        reasoning = prediction.reasoning

        # Should contain the flag
        assert "[LOW CONFIDENCE:" in reasoning, (
            f"Low confidence prediction should contain flag. "
            f"Reasoning: '{reasoning}'"
        )

        # Should end with closing bracket
        assert "]" in reasoning, (
            f"Low confidence flag should be properly closed with ]. "
            f"Reasoning: '{reasoning}'"
        )

        # Should contain the confidence value formatted to 2 decimal places
        confidence_str = f"{prediction.confidence_level:.2f}"
        assert confidence_str in reasoning, (
            f"Low confidence flag should contain confidence value {confidence_str}. "
            f"Reasoning: '{reasoning}'"
        )


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_stock_strategy(),
)
@settings(max_examples=100)
def test_flagging_does_not_affect_other_fields(article, sentiment, entities):
    """
    **Feature: news-market-predictor, Property 13: Low confidence flagging**

    Property: Low confidence flagging should only affect the reasoning field,
    not other prediction fields.

    **Validates: Requirements 3.5**

    This test verifies that flagging low confidence predictions doesn't
    corrupt other prediction data.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Initialize predictor
    predictor = BasicMarketPredictor(historical_accuracy=0.65)

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Skip test if no predictions generated
    assume(len(predictions) > 0)

    # Verify that flagging doesn't affect other fields
    for prediction in predictions:
        # All predictions should have valid basic fields regardless of confidence
        assert prediction.article_id == article.id
        assert prediction.stock_symbol is not None and prediction.stock_symbol != ""
        assert prediction.impact_direction in ["positive", "negative", "neutral"]
        assert 0.0 <= prediction.impact_magnitude <= 1.0
        assert 0.0 <= prediction.confidence_level <= 1.0
        assert isinstance(prediction.reasoning, str)
        assert prediction.created_at is not None

        # Low confidence flagging should only add to reasoning, not replace it
        if "[LOW CONFIDENCE:" in prediction.reasoning:
            # Should still have some reasoning before the flag
            flag_index = prediction.reasoning.find("[LOW CONFIDENCE:")
            reasoning_before_flag = prediction.reasoning[:flag_index].strip()
            assert len(reasoning_before_flag) > 0, (
                f"Low confidence flag should be added to existing reasoning, "
                f"not replace it. Reasoning: '{prediction.reasoning}'"
            )
