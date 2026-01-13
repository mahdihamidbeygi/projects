"""
Property-based tests for multi-stock prediction completeness.

**Feature: news-market-predictor, Property 11: Multi-stock prediction completeness**
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


# Strategy for generating multiple stock symbols
@st.composite
def multiple_stock_entities_strategy(draw, article_id=None, min_stocks=2, max_stocks=5):
    """Generate multiple ExtractedEntity objects with stock_symbol type for testing."""
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
        "CRM",
        "ORCL",
        "IBM",
        "ADBE",
        "PYPL",
        "UBER",
        "LYFT",
        "SNAP",
        "TWTR",
        "SPOT",
    ]

    # Generate multiple unique stock symbols
    num_stocks = draw(st.integers(min_value=min_stocks, max_value=max_stocks))
    selected_symbols = draw(
        st.lists(
            st.sampled_from(stock_symbols),
            min_size=num_stocks,
            max_size=num_stocks,
            unique=True,
        )
    )

    stock_entities = []
    for symbol in selected_symbols:
        entity = ExtractedEntity(
            article_id=article_id,
            entity_type="stock_symbol",
            entity_value=symbol,
            relevance_score=draw(
                st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                )
            ),
            context=draw(st.text(min_size=1, max_size=200)),
        )
        stock_entities.append(entity)

    return stock_entities


# Strategy for generating entities with multiple stocks plus other entity types
@st.composite
def entities_with_multiple_stocks_strategy(draw, article_id=None):
    """Generate list of entities that includes multiple stock symbols and other entities."""
    if article_id is None:
        article_id = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            )
        )

    # Always include multiple stock symbols
    stock_entities = draw(multiple_stock_entities_strategy(article_id))
    entities = stock_entities.copy()

    # Optionally add other entity types
    additional_entities = draw(
        st.lists(
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
            min_size=0,
            max_size=3,
        )
    )

    entities.extend(additional_entities)
    return entities


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_multiple_stocks_strategy(),
)
@settings(max_examples=100)
def test_multi_stock_prediction_completeness_for_any_input(
    article, sentiment, entities
):
    """
    **Feature: news-market-predictor, Property 11: Multi-stock prediction completeness**

    Property: For any article mentioning multiple stock symbols, separate predictions
    should be generated for each identified symbol.

    **Validates: Requirements 3.3**

    This test verifies that when multiple stock symbols are present in the entities,
    the market predictor generates a separate prediction for each stock symbol.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Get the stock symbols from entities
    stock_entities = [e for e in entities if e.entity_type == "stock_symbol"]
    stock_symbols = [e.entity_value for e in stock_entities]

    # Ensure we have multiple stock symbols (this should be guaranteed by the strategy)
    assume(len(stock_symbols) >= 2)

    # Initialize the market predictor
    predictor = BasicMarketPredictor()

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Verify that predictions were generated for each stock symbol
    assert len(predictions) >= len(stock_symbols), (
        f"Should generate at least {len(stock_symbols)} predictions for "
        f"{len(stock_symbols)} stock symbols, but got {len(predictions)}"
    )

    # Verify each stock symbol has at least one prediction
    predicted_symbols = {p.stock_symbol for p in predictions}
    for stock_symbol in stock_symbols:
        assert stock_symbol in predicted_symbols, (
            f"Missing prediction for stock symbol '{stock_symbol}'. "
            f"Expected symbols: {stock_symbols}, Got predictions for: {predicted_symbols}"
        )

    # Verify each prediction is valid
    for prediction in predictions:
        assert isinstance(prediction, MarketPrediction)
        assert prediction.stock_symbol in stock_symbols
        assert prediction.article_id == article.id
        assert prediction.validate() is True


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
)
@settings(max_examples=100)
def test_multi_stock_prediction_completeness_with_exact_count(article, sentiment):
    """
    **Feature: news-market-predictor, Property 11: Multi-stock prediction completeness**

    Property: The number of predictions should match the number of unique stock symbols
    when no errors occur during prediction generation.

    **Validates: Requirements 3.3**

    This test verifies that exactly one prediction is generated per stock symbol
    under normal conditions.
    """
    # Ensure sentiment has matching article_id
    sentiment.article_id = article.id

    # Create exactly 3 unique stock symbols
    stock_entities = [
        ExtractedEntity(
            article_id=article.id,
            entity_type="stock_symbol",
            entity_value="AAPL",
            relevance_score=0.8,
            context="Apple mentioned in earnings report",
        ),
        ExtractedEntity(
            article_id=article.id,
            entity_type="stock_symbol",
            entity_value="GOOGL",
            relevance_score=0.7,
            context="Google discussed in tech analysis",
        ),
        ExtractedEntity(
            article_id=article.id,
            entity_type="stock_symbol",
            entity_value="MSFT",
            relevance_score=0.9,
            context="Microsoft featured in market update",
        ),
    ]

    # Add some non-stock entities
    other_entities = [
        ExtractedEntity(
            article_id=article.id,
            entity_type="company",
            entity_value="Tesla Inc",
            relevance_score=0.6,
            context="Company name mentioned",
        ),
        ExtractedEntity(
            article_id=article.id,
            entity_type="metric",
            entity_value="revenue",
            relevance_score=0.5,
            context="Financial metric discussed",
        ),
    ]

    entities = stock_entities + other_entities

    # Initialize the market predictor
    predictor = BasicMarketPredictor()

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Should have exactly 3 predictions (one per stock symbol)
    assert (
        len(predictions) == 3
    ), f"Expected exactly 3 predictions for 3 stock symbols, got {len(predictions)}"

    # Verify each stock symbol has exactly one prediction
    predicted_symbols = [p.stock_symbol for p in predictions]
    expected_symbols = ["AAPL", "GOOGL", "MSFT"]

    for symbol in expected_symbols:
        count = predicted_symbols.count(symbol)
        assert count == 1, f"Expected exactly 1 prediction for {symbol}, got {count}"


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_multiple_stocks_strategy(),
)
@settings(max_examples=100)
def test_multi_stock_prediction_completeness_with_duplicates(
    article, sentiment, entities
):
    """
    **Feature: news-market-predictor, Property 11: Multi-stock prediction completeness**

    Property: When duplicate stock symbols exist in entities, predictions should still
    be generated appropriately (one per unique symbol).

    **Validates: Requirements 3.3**

    This test verifies proper handling of duplicate stock symbols in the entity list.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Add duplicate stock entities
    stock_entities = [e for e in entities if e.entity_type == "stock_symbol"]
    if stock_entities:
        # Duplicate the first stock entity
        duplicate_entity = ExtractedEntity(
            article_id=article.id,
            entity_type="stock_symbol",
            entity_value=stock_entities[0].entity_value,  # Same symbol
            relevance_score=0.6,
            context="Duplicate mention of the same stock",
        )
        entities.append(duplicate_entity)

    # Get unique stock symbols
    stock_symbols = list(
        {e.entity_value for e in entities if e.entity_type == "stock_symbol"}
    )

    # Ensure we have multiple unique stock symbols
    assume(len(stock_symbols) >= 2)

    # Initialize the market predictor
    predictor = BasicMarketPredictor()

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Verify predictions cover all unique stock symbols
    predicted_symbols = {p.stock_symbol for p in predictions}

    for stock_symbol in stock_symbols:
        assert stock_symbol in predicted_symbols, (
            f"Missing prediction for unique stock symbol '{stock_symbol}'. "
            f"Expected symbols: {stock_symbols}, Got predictions for: {predicted_symbols}"
        )

    # Verify no extra predictions beyond the unique symbols
    for predicted_symbol in predicted_symbols:
        assert (
            predicted_symbol in stock_symbols
        ), f"Unexpected prediction for symbol '{predicted_symbol}' not in input entities"


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=entities_with_multiple_stocks_strategy(),
)
@settings(max_examples=100)
def test_multi_stock_prediction_completeness_consistency(article, sentiment, entities):
    """
    **Feature: news-market-predictor, Property 11: Multi-stock prediction completeness**

    Property: Multiple calls with the same input should consistently generate
    predictions for all stock symbols.

    **Validates: Requirements 3.3**

    This test verifies that multi-stock prediction generation is deterministic
    and consistent across multiple calls.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Get stock symbols
    stock_symbols = [
        e.entity_value for e in entities if e.entity_type == "stock_symbol"
    ]
    assume(len(stock_symbols) >= 2)

    # Initialize the market predictor
    predictor = BasicMarketPredictor()

    # Generate predictions multiple times
    predictions1 = predictor.predict_impact(article, sentiment, entities)
    predictions2 = predictor.predict_impact(article, sentiment, entities)
    predictions3 = predictor.predict_impact(article, sentiment, entities)

    # All prediction sets should have the same length
    assert (
        len(predictions1) == len(predictions2) == len(predictions3)
    ), "Prediction count should be consistent across multiple calls"

    # All prediction sets should cover the same stock symbols
    symbols1 = {p.stock_symbol for p in predictions1}
    symbols2 = {p.stock_symbol for p in predictions2}
    symbols3 = {p.stock_symbol for p in predictions3}

    assert (
        symbols1 == symbols2 == symbols3
    ), "Predicted stock symbols should be consistent across multiple calls"

    # All stock symbols should be covered in each call
    for stock_symbol in stock_symbols:
        assert (
            stock_symbol in symbols1
        ), f"Stock symbol '{stock_symbol}' missing from predictions"


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
)
@settings(max_examples=100)
def test_multi_stock_prediction_completeness_with_varying_relevance(article, sentiment):
    """
    **Feature: news-market-predictor, Property 11: Multi-stock prediction completeness**

    Property: Predictions should be generated for all stock symbols regardless
    of their relevance scores.

    **Validates: Requirements 3.3**

    This test verifies that low relevance scores don't prevent prediction generation
    for any mentioned stock symbols.
    """
    # Ensure sentiment has matching article_id
    sentiment.article_id = article.id

    # Create stock entities with varying relevance scores
    stock_entities = [
        ExtractedEntity(
            article_id=article.id,
            entity_type="stock_symbol",
            entity_value="AAPL",
            relevance_score=0.9,  # High relevance
            context="Apple prominently featured",
        ),
        ExtractedEntity(
            article_id=article.id,
            entity_type="stock_symbol",
            entity_value="GOOGL",
            relevance_score=0.1,  # Low relevance
            context="Google briefly mentioned",
        ),
        ExtractedEntity(
            article_id=article.id,
            entity_type="stock_symbol",
            entity_value="MSFT",
            relevance_score=0.5,  # Medium relevance
            context="Microsoft discussed in context",
        ),
    ]

    # Initialize the market predictor
    predictor = BasicMarketPredictor()

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, stock_entities)

    # Should generate predictions for all stocks regardless of relevance
    predicted_symbols = {p.stock_symbol for p in predictions}
    expected_symbols = {"AAPL", "GOOGL", "MSFT"}

    assert predicted_symbols == expected_symbols, (
        f"Should generate predictions for all stock symbols regardless of relevance. "
        f"Expected: {expected_symbols}, Got: {predicted_symbols}"
    )

    # Verify all predictions are valid
    for prediction in predictions:
        assert isinstance(prediction, MarketPrediction)
        assert prediction.validate() is True


@given(
    article=news_article_strategy(),
    sentiment=sentiment_analysis_strategy(),
    entities=multiple_stock_entities_strategy(min_stocks=5, max_stocks=10),
)
@settings(max_examples=100)
def test_multi_stock_prediction_completeness_with_many_stocks(
    article, sentiment, entities
):
    """
    **Feature: news-market-predictor, Property 11: Multi-stock prediction completeness**

    Property: The system should handle articles with many stock symbols and generate
    predictions for all of them.

    **Validates: Requirements 3.3**

    This test verifies scalability when dealing with articles mentioning many stocks.
    """
    # Ensure sentiment and entities have matching article_id
    sentiment.article_id = article.id
    for entity in entities:
        entity.article_id = article.id

    # Get stock symbols (should be 5-10 based on strategy)
    stock_symbols = [
        e.entity_value for e in entities if e.entity_type == "stock_symbol"
    ]
    assume(len(stock_symbols) >= 5)

    # Initialize the market predictor
    predictor = BasicMarketPredictor()

    # Generate predictions
    predictions = predictor.predict_impact(article, sentiment, entities)

    # Should generate predictions for all stock symbols
    predicted_symbols = {p.stock_symbol for p in predictions}

    assert len(predicted_symbols) >= len(stock_symbols), (
        f"Should generate predictions for all {len(stock_symbols)} stock symbols, "
        f"but only got predictions for {len(predicted_symbols)} symbols"
    )

    # Verify each stock symbol has a prediction
    for stock_symbol in stock_symbols:
        assert (
            stock_symbol in predicted_symbols
        ), f"Missing prediction for stock symbol '{stock_symbol}' in large stock set"

    # Verify all predictions are valid
    for prediction in predictions:
        assert isinstance(prediction, MarketPrediction)
        assert prediction.stock_symbol in stock_symbols
        assert prediction.validate() is True
