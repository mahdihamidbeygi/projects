"""
Property-based tests for display output completeness.

**Feature: news-market-predictor, Property 14: Display output completeness**
"""

from datetime import datetime
from hypothesis import given, strategies as st, settings

from news_market_predictor.aggregator.display_formatter import DisplayFormatter
from news_market_predictor.models import (
    NewsArticle,
    SentimentAnalysis,
    ExtractedEntity,
    MarketPrediction,
    HistoricalAccuracy,
)


# Strategy for generating valid NewsArticle objects
@st.composite
def news_article_strategy(draw):
    """Generate valid NewsArticle objects for testing."""
    return NewsArticle(
        id=draw(
            st.text(
                min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"
            )
        ),
        title=draw(
            st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz ")
        ),
        content=draw(
            st.text(min_size=10, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz ")
        ),
        url="https://finance.yahoo.com/news/test-article",
        published_at=datetime(2024, 1, 1),
        source=draw(st.sampled_from(["Yahoo Finance", "Reuters", "Bloomberg"])),
        category=draw(st.sampled_from(["earnings", "markets", "technology"])),
        raw_metadata={},
    )


# Strategy for generating valid MarketPrediction objects
@st.composite
def market_prediction_strategy(draw, article_id=None, stock_symbol=None):
    """Generate valid MarketPrediction objects for testing."""
    if article_id is None:
        article_id = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            )
        )

    if stock_symbol is None:
        stock_symbol = draw(
            st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"])
        )

    return MarketPrediction(
        article_id=article_id,
        stock_symbol=stock_symbol,
        impact_direction=draw(st.sampled_from(["positive", "negative", "neutral"])),
        impact_magnitude=draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        confidence_level=draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        reasoning=draw(st.text(min_size=1, max_size=500)),
        created_at=draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
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


# Strategy for generating valid ExtractedEntity objects
@st.composite
def extracted_entity_strategy(draw, article_id=None):
    """Generate valid ExtractedEntity objects for testing."""
    if article_id is None:
        article_id = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            )
        )

    return ExtractedEntity(
        article_id=article_id,
        entity_type=draw(st.sampled_from(["stock_symbol", "company", "metric"])),
        entity_value=draw(st.text(min_size=1, max_size=50)),
        relevance_score=draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        context=draw(st.text(min_size=1, max_size=200)),
    )


# Strategy for generating valid HistoricalAccuracy objects
@st.composite
def historical_accuracy_strategy(draw, stock_symbol=None):
    """Generate valid HistoricalAccuracy objects for testing."""
    if stock_symbol is None:
        stock_symbol = draw(
            st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"])
        )

    total_predictions = draw(st.integers(min_value=1, max_value=1000))
    correct_predictions = draw(st.integers(min_value=0, max_value=total_predictions))
    accuracy_rate = (
        correct_predictions / total_predictions if total_predictions > 0 else 0.0
    )

    return HistoricalAccuracy(
        stock_symbol=stock_symbol,
        time_period_days=draw(st.integers(min_value=1, max_value=365)),
        total_predictions=total_predictions,
        correct_predictions=correct_predictions,
        accuracy_rate=accuracy_rate,
        average_confidence=draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        calculated_at=draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
            )
        ),
    )


@given(
    prediction=market_prediction_strategy(),
    article=news_article_strategy(),
    entities=st.lists(extracted_entity_strategy(), min_size=0, max_size=10),
    sentiment=sentiment_analysis_strategy(),
    historical_accuracy=historical_accuracy_strategy(),
)
@settings(max_examples=100)
def test_display_output_completeness_all_required_fields_present(
    prediction, article, entities, sentiment, historical_accuracy
):
    """
    **Feature: news-market-predictor, Property 14: Display output completeness**

    Property: For any prediction display, all required fields (impact, confidence,
    stock symbols, article title) should be present.

    **Validates: Requirements 4.1, 4.2**

    This test verifies that the display formatter includes all required fields
    as specified in requirements 4.1 and 4.2.
    """
    # Ensure all objects have matching IDs
    article.id = prediction.article_id
    sentiment.article_id = prediction.article_id
    historical_accuracy.stock_symbol = prediction.stock_symbol
    for entity in entities:
        entity.article_id = prediction.article_id

    # Initialize the display formatter
    formatter = DisplayFormatter()

    # Format the prediction for display
    display_output = formatter.format_prediction_display(
        prediction=prediction,
        article=article,
        entities=entities,
        sentiment=sentiment,
        historical_accuracy=historical_accuracy,
    )

    # Verify display output is a dictionary
    assert isinstance(display_output, dict), "Display output must be a dictionary"

    # Verify core prediction fields are present (Requirement 4.1)
    # Impact prediction with direction, magnitude, and confidence
    assert "impact_prediction" in display_output, "Impact prediction must be present"
    impact_pred = display_output["impact_prediction"]
    assert isinstance(impact_pred, dict), "Impact prediction must be a dictionary"

    assert "direction" in impact_pred, "Impact direction must be present"
    assert "magnitude" in impact_pred, "Impact magnitude must be present"
    assert "confidence_level" in impact_pred, "Confidence level must be present"

    # Stock symbol
    assert "stock_symbol" in display_output, "Stock symbol must be present"
    assert display_output["stock_symbol"] == prediction.stock_symbol

    # Verify article information is present (Requirement 4.2)
    assert "article_info" in display_output, "Article information must be present"
    article_info = display_output["article_info"]
    assert isinstance(article_info, dict), "Article info must be a dictionary"

    # Article title (specifically mentioned in requirements)
    assert "title" in article_info, "Article title must be present"
    assert article_info["title"] == article.title

    # Key extracted information (Requirement 4.2)
    assert "extracted_entities" in display_output, "Extracted entities must be present"
    extracted_info = display_output["extracted_entities"]
    assert isinstance(extracted_info, dict), "Extracted entities must be a dictionary"

    # Should have categories for different entity types
    assert "stock_symbols" in extracted_info, "Stock symbols section must be present"
    assert "companies" in extracted_info, "Companies section must be present"
    assert (
        "financial_metrics" in extracted_info
    ), "Financial metrics section must be present"

    # Verify all required fields have appropriate data types
    assert isinstance(display_output["stock_symbol"], str)
    assert isinstance(impact_pred["direction"], str)
    assert isinstance(impact_pred["magnitude"], (int, float))
    assert isinstance(impact_pred["confidence_level"], (int, float))
    assert isinstance(article_info["title"], str)
    assert isinstance(extracted_info["stock_symbols"], list)
    assert isinstance(extracted_info["companies"], list)
    assert isinstance(extracted_info["financial_metrics"], list)


@given(prediction=market_prediction_strategy())
@settings(max_examples=100)
def test_display_output_completeness_minimal_required_fields(prediction):
    """
    **Feature: news-market-predictor, Property 14: Display output completeness**

    Property: Even with minimal input (just prediction), core required fields
    should still be present in display output.

    **Validates: Requirements 4.1, 4.2**

    This test verifies that essential fields are present even when optional
    data is not provided.
    """
    # Initialize the display formatter
    formatter = DisplayFormatter()

    # Format the prediction with minimal data (no article, entities, etc.)
    display_output = formatter.format_prediction_display(prediction=prediction)

    # Verify core required fields are still present
    assert isinstance(display_output, dict), "Display output must be a dictionary"

    # Core prediction data must always be present (Requirement 4.1)
    assert "impact_prediction" in display_output, "Impact prediction must be present"
    impact_pred = display_output["impact_prediction"]

    assert "direction" in impact_pred, "Impact direction must be present"
    assert "magnitude" in impact_pred, "Impact magnitude must be present"
    assert "confidence_level" in impact_pred, "Confidence level must be present"

    assert "stock_symbol" in display_output, "Stock symbol must be present"
    assert display_output["stock_symbol"] == prediction.stock_symbol

    # Verify values match the prediction
    assert impact_pred["direction"] == prediction.impact_direction
    assert impact_pred["magnitude"] == prediction.impact_magnitude
    assert impact_pred["confidence_level"] == prediction.confidence_level


@given(
    predictions=st.lists(market_prediction_strategy(), min_size=1, max_size=10),
)
@settings(max_examples=100)
def test_display_output_completeness_aggregated_predictions(predictions):
    """
    **Feature: news-market-predictor, Property 14: Display output completeness**

    Property: For any list of predictions, each formatted prediction in the
    aggregated display should contain all required fields.

    **Validates: Requirements 4.1, 4.2**

    This test verifies that aggregated prediction displays maintain completeness
    for each individual prediction.
    """
    # Initialize the display formatter
    formatter = DisplayFormatter()

    # Format aggregated predictions
    display_outputs = formatter.format_aggregated_predictions(predictions=predictions)

    # Verify we get a list of formatted predictions
    assert isinstance(display_outputs, list), "Aggregated output must be a list"
    assert len(display_outputs) == len(predictions), "Should format all predictions"

    # Verify each formatted prediction has required fields
    for i, display_output in enumerate(display_outputs):
        original_prediction = predictions[i]

        # Verify it's a dictionary
        assert isinstance(
            display_output, dict
        ), f"Display output {i} must be a dictionary"

        # Core prediction fields (Requirement 4.1)
        assert (
            "impact_prediction" in display_output
        ), f"Impact prediction must be present in output {i}"
        impact_pred = display_output["impact_prediction"]

        assert (
            "direction" in impact_pred
        ), f"Impact direction must be present in output {i}"
        assert (
            "magnitude" in impact_pred
        ), f"Impact magnitude must be present in output {i}"
        assert (
            "confidence_level" in impact_pred
        ), f"Confidence level must be present in output {i}"

        assert (
            "stock_symbol" in display_output
        ), f"Stock symbol must be present in output {i}"

        # Verify values match original prediction
        assert impact_pred["direction"] == original_prediction.impact_direction
        assert impact_pred["magnitude"] == original_prediction.impact_magnitude
        assert impact_pred["confidence_level"] == original_prediction.confidence_level
        assert display_output["stock_symbol"] == original_prediction.stock_symbol

        # Aggregation info should be present
        assert (
            "aggregation_info" in display_output
        ), f"Aggregation info must be present in output {i}"
        assert isinstance(display_output["aggregation_info"], dict)
        assert "is_aggregated" in display_output["aggregation_info"]


@given(
    prediction=market_prediction_strategy(),
    article=news_article_strategy(),
    entities=st.lists(extracted_entity_strategy(), min_size=1, max_size=5),
)
@settings(max_examples=100)
def test_display_output_completeness_key_extracted_information(
    prediction, article, entities
):
    """
    **Feature: news-market-predictor, Property 14: Display output completeness**

    Property: When entities are provided, the display should include key extracted
    information organized by entity type.

    **Validates: Requirements 4.2**

    This test specifically verifies that key extracted information is properly
    included and organized in the display output.
    """
    # Ensure all objects have matching IDs
    article.id = prediction.article_id
    for entity in entities:
        entity.article_id = prediction.article_id

    # Initialize the display formatter
    formatter = DisplayFormatter()

    # Format the prediction for display
    display_output = formatter.format_prediction_display(
        prediction=prediction,
        article=article,
        entities=entities,
    )

    # Verify extracted entities section is present and properly structured
    assert "extracted_entities" in display_output, "Extracted entities must be present"
    extracted_info = display_output["extracted_entities"]
    assert isinstance(extracted_info, dict), "Extracted entities must be a dictionary"

    # Verify all entity type categories are present
    required_categories = ["stock_symbols", "companies", "financial_metrics"]
    for category in required_categories:
        assert category in extracted_info, f"{category} category must be present"
        assert isinstance(extracted_info[category], list), f"{category} must be a list"

    # Verify entities are properly categorized
    stock_entities = [e for e in entities if e.entity_type == "stock_symbol"]
    company_entities = [e for e in entities if e.entity_type == "company"]
    metric_entities = [e for e in entities if e.entity_type == "metric"]

    # Check that entities are properly placed in their categories
    assert len(extracted_info["stock_symbols"]) == len(stock_entities)
    assert len(extracted_info["companies"]) == len(company_entities)
    assert len(extracted_info["financial_metrics"]) == len(metric_entities)

    # Verify each entity in the display has required fields
    for stock_entity in extracted_info["stock_symbols"]:
        assert "symbol" in stock_entity, "Stock entity must have symbol field"
        assert "relevance" in stock_entity, "Stock entity must have relevance field"
        assert isinstance(stock_entity["symbol"], str)
        assert isinstance(stock_entity["relevance"], (int, float))

    for company_entity in extracted_info["companies"]:
        assert "name" in company_entity, "Company entity must have name field"
        assert "relevance" in company_entity, "Company entity must have relevance field"
        assert isinstance(company_entity["name"], str)
        assert isinstance(company_entity["relevance"], (int, float))

    for metric_entity in extracted_info["financial_metrics"]:
        assert "metric" in metric_entity, "Metric entity must have metric field"
        assert "context" in metric_entity, "Metric entity must have context field"
        assert "relevance" in metric_entity, "Metric entity must have relevance field"
        assert isinstance(metric_entity["metric"], str)
        assert isinstance(metric_entity["context"], str)
        assert isinstance(metric_entity["relevance"], (int, float))


@given(
    prediction=market_prediction_strategy(),
    article=news_article_strategy(),
)
@settings(max_examples=100)
def test_display_output_completeness_article_title_always_present(prediction, article):
    """
    **Feature: news-market-predictor, Property 14: Display output completeness**

    Property: When an article is provided, the article title must always be
    present in the display output.

    **Validates: Requirements 4.2**

    This test specifically verifies that the article title, which is explicitly
    mentioned in requirement 4.2, is always included when available.
    """
    # Ensure matching IDs
    article.id = prediction.article_id

    # Initialize the display formatter
    formatter = DisplayFormatter()

    # Format the prediction for display
    display_output = formatter.format_prediction_display(
        prediction=prediction,
        article=article,
    )

    # Verify article information is present
    assert "article_info" in display_output, "Article information must be present"
    article_info = display_output["article_info"]

    # Verify article title is specifically present (Requirement 4.2)
    assert "title" in article_info, "Article title must be present"
    assert article_info["title"] == article.title, "Article title must match original"
    assert isinstance(article_info["title"], str), "Article title must be a string"
    assert len(article_info["title"]) > 0, "Article title must not be empty"

    # Verify other article fields are also present for completeness
    expected_article_fields = ["title", "published_at", "source", "category", "url"]
    for field in expected_article_fields:
        assert field in article_info, f"Article {field} must be present"
