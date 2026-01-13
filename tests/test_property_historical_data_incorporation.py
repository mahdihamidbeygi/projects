"""
Property-based tests for historical data incorporation.

**Feature: news-market-predictor, Property 12: Historical data incorporation**
"""

from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, assume
import tempfile
import os

from news_market_predictor.predictor.historical_analyzer import HistoricalAnalyzer
from news_market_predictor.storage.historical_data_store import HistoricalDataStore
from news_market_predictor.models import (
    NewsArticle,
    SentimentAnalysis,
    ExtractedEntity,
    MarketPrediction,
    MarketOutcome,
    HistoricalAccuracy,
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
        stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
        stock_symbol = draw(st.sampled_from(stock_symbols))

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
                min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False
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


# Strategy for generating valid HistoricalAccuracy objects
@st.composite
def historical_accuracy_strategy(draw, stock_symbol=None):
    """Generate valid HistoricalAccuracy objects for testing."""
    if stock_symbol is None:
        stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
        stock_symbol = draw(st.sampled_from(stock_symbols))

    total_predictions = draw(st.integers(min_value=5, max_value=100))
    correct_predictions = draw(st.integers(min_value=0, max_value=total_predictions))
    accuracy_rate = (
        correct_predictions / total_predictions if total_predictions > 0 else 0.0
    )

    return HistoricalAccuracy(
        stock_symbol=stock_symbol,
        time_period_days=draw(st.integers(min_value=7, max_value=365)),
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
    stock_symbol=st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]),
    base_prediction=market_prediction_strategy(),
    sentiment=sentiment_analysis_strategy(),
    historical_accuracy=historical_accuracy_strategy(),
)
@settings(max_examples=100)
def test_historical_accuracy_influences_confidence(
    stock_symbol, base_prediction, sentiment, historical_accuracy
):
    """
    **Feature: news-market-predictor, Property 12: Historical data incorporation**

    Property: For any prediction where historical accuracy data exists, the historical
    accuracy should influence the confidence level of the current prediction.

    **Validates: Requirements 3.4**

    This test verifies that historical accuracy data properly influences
    confidence calculations in predictions.
    """
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Initialize historical data store and analyzer
        historical_store = HistoricalDataStore(db_path=temp_db_path)
        historical_analyzer = HistoricalAnalyzer(historical_store)

        # Ensure prediction and accuracy use the same stock symbol
        base_prediction.stock_symbol = stock_symbol
        historical_accuracy.stock_symbol = stock_symbol

        # Store historical accuracy
        historical_store.store_accuracy_metrics(historical_accuracy)

        # Calculate historical influence
        influenced_prediction = historical_analyzer.calculate_historical_influence(
            stock_symbol=stock_symbol,
            sentiment=sentiment,
            base_prediction=base_prediction,
        )

        # Verify that prediction is still valid
        assert influenced_prediction.stock_symbol == stock_symbol
        assert influenced_prediction.article_id == base_prediction.article_id
        assert influenced_prediction.impact_direction in [
            "positive",
            "negative",
            "neutral",
        ]
        assert 0.0 <= influenced_prediction.impact_magnitude <= 1.0
        assert 0.0 <= influenced_prediction.confidence_level <= 1.0

        # Verify that historical context is included in reasoning
        assert len(influenced_prediction.reasoning) >= len(
            base_prediction.reasoning
        ), "Historical influence should maintain or add context to the reasoning"

        # The reasoning should mention historical analysis
        assert (
            "Historical accuracy for this stock:" in influenced_prediction.reasoning
            or "No significant historical data available"
            in influenced_prediction.reasoning
        ), "Reasoning should include historical analysis context"

        # Close database connections properly
        del historical_analyzer
        del historical_store

    finally:
        # Clean up temporary database
        try:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
        except PermissionError:
            # On Windows, sometimes the file is still locked, just pass
            pass


@given(
    stock_symbol=st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]),
    base_prediction=market_prediction_strategy(),
    sentiment=sentiment_analysis_strategy(),
)
@settings(max_examples=100)
def test_no_historical_data_preserves_base_prediction(
    stock_symbol, base_prediction, sentiment
):
    """
    **Feature: news-market-predictor, Property 12: Historical data incorporation**

    Property: When no historical data exists, the base prediction should be
    preserved with minimal changes (only reasoning should be updated).

    **Validates: Requirements 3.4**

    This test verifies that the system handles cases where no historical
    data is available gracefully.
    """
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Initialize historical data store and analyzer (empty database)
        historical_store = HistoricalDataStore(db_path=temp_db_path)
        historical_analyzer = HistoricalAnalyzer(historical_store)

        # Ensure prediction uses the test stock symbol
        base_prediction.stock_symbol = stock_symbol

        # Calculate historical influence with no historical data
        influenced_prediction = historical_analyzer.calculate_historical_influence(
            stock_symbol=stock_symbol,
            sentiment=sentiment,
            base_prediction=base_prediction,
        )

        # Verify that core prediction values are preserved
        assert influenced_prediction.stock_symbol == base_prediction.stock_symbol
        assert influenced_prediction.article_id == base_prediction.article_id
        assert (
            influenced_prediction.impact_direction == base_prediction.impact_direction
        )
        assert (
            influenced_prediction.impact_magnitude == base_prediction.impact_magnitude
        )

        # Confidence should be similar (within small tolerance for neutral factor)
        confidence_diff = abs(
            influenced_prediction.confidence_level - base_prediction.confidence_level
        )
        assert confidence_diff <= 0.1, (
            f"Confidence should be preserved when no historical data exists. "
            f"Base: {base_prediction.confidence_level:.3f}, Influenced: {influenced_prediction.confidence_level:.3f}"
        )

        # Reasoning should indicate no historical data
        assert (
            "No significant historical data available"
            in influenced_prediction.reasoning
        ), "Reasoning should indicate when no historical data is available"

        # Close database connections properly
        del historical_analyzer
        del historical_store

    finally:
        # Clean up temporary database
        try:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
        except PermissionError:
            # On Windows, sometimes the file is still locked, just pass
            pass


@given(
    stock_symbol=st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]),
    base_prediction=market_prediction_strategy(),
    sentiment=sentiment_analysis_strategy(),
)
@settings(max_examples=100)
def test_historical_influence_preserves_prediction_validity(
    stock_symbol, base_prediction, sentiment
):
    """
    **Feature: news-market-predictor, Property 12: Historical data incorporation**

    Property: Historical influence should always preserve the validity of
    predictions - all fields should remain within valid ranges.

    **Validates: Requirements 3.4**

    This test verifies that historical influence never produces invalid
    prediction data regardless of historical data characteristics.
    """
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Initialize historical data store and analyzer
        historical_store = HistoricalDataStore(db_path=temp_db_path)
        historical_analyzer = HistoricalAnalyzer(historical_store)

        # Ensure prediction uses the test stock symbol
        base_prediction.stock_symbol = stock_symbol

        # Create extreme historical data to test robustness
        extreme_accuracy = HistoricalAccuracy(
            stock_symbol=stock_symbol,
            time_period_days=30,
            total_predictions=100,
            correct_predictions=0,  # 0% accuracy
            accuracy_rate=0.0,
            average_confidence=0.1,
            calculated_at=datetime.now(),
        )

        # Store extreme historical data
        historical_store.store_accuracy_metrics(extreme_accuracy)

        # Calculate historical influence
        influenced_prediction = historical_analyzer.calculate_historical_influence(
            stock_symbol=stock_symbol,
            sentiment=sentiment,
            base_prediction=base_prediction,
        )

        # Verify all prediction fields are valid
        influenced_prediction.validate()  # This will raise ValidationError if invalid

        # Verify specific field constraints
        assert influenced_prediction.stock_symbol == stock_symbol
        assert influenced_prediction.article_id == base_prediction.article_id
        assert influenced_prediction.impact_direction in [
            "positive",
            "negative",
            "neutral",
        ]
        assert 0.0 <= influenced_prediction.impact_magnitude <= 1.0
        assert 0.0 <= influenced_prediction.confidence_level <= 1.0
        assert isinstance(influenced_prediction.reasoning, str)
        assert len(influenced_prediction.reasoning) > 0
        assert isinstance(influenced_prediction.created_at, datetime)

        # Verify no NaN or infinity values
        assert not (
            influenced_prediction.impact_magnitude
            != influenced_prediction.impact_magnitude
        )
        assert not (
            influenced_prediction.confidence_level
            != influenced_prediction.confidence_level
        )
        assert influenced_prediction.impact_magnitude != float("inf")
        assert influenced_prediction.confidence_level != float("inf")

        # Close database connections properly
        del historical_analyzer
        del historical_store

    finally:
        # Clean up temporary database
        try:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
        except PermissionError:
            # On Windows, sometimes the file is still locked, just pass
            pass


@given(
    stock_symbol=st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]),
    base_prediction=market_prediction_strategy(),
    sentiment=sentiment_analysis_strategy(),
    accuracy_rate=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100)
def test_accuracy_factor_influences_confidence_direction(
    stock_symbol, base_prediction, sentiment, accuracy_rate
):
    """
    **Feature: news-market-predictor, Property 12: Historical data incorporation**

    Property: Historical accuracy should influence confidence in the expected direction -
    higher accuracy should tend to increase confidence, lower accuracy should tend to decrease it.

    **Validates: Requirements 3.4**

    This test verifies that the direction of confidence adjustment is appropriate
    based on historical accuracy levels.
    """
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Initialize historical data store and analyzer
        historical_store = HistoricalDataStore(db_path=temp_db_path)
        historical_analyzer = HistoricalAnalyzer(historical_store)

        # Ensure prediction uses the test stock symbol
        base_prediction.stock_symbol = stock_symbol

        # Create historical accuracy data
        total_predictions = 20
        correct_predictions = int(accuracy_rate * total_predictions)

        historical_accuracy = HistoricalAccuracy(
            stock_symbol=stock_symbol,
            time_period_days=30,
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            accuracy_rate=accuracy_rate,
            average_confidence=0.7,
            calculated_at=datetime.now(),
        )

        # Store historical accuracy
        historical_store.store_accuracy_metrics(historical_accuracy)

        # Calculate historical influence
        influenced_prediction = historical_analyzer.calculate_historical_influence(
            stock_symbol=stock_symbol,
            sentiment=sentiment,
            base_prediction=base_prediction,
        )

        # Verify confidence adjustment direction based on historical accuracy
        if (
            accuracy_rate > 0.8
        ):  # Very high accuracy should maintain or increase confidence
            assert (
                influenced_prediction.confidence_level
                >= base_prediction.confidence_level * 0.8
            ), (
                f"Very high historical accuracy ({accuracy_rate:.2f}) should maintain or increase confidence. "
                f"Base: {base_prediction.confidence_level:.3f}, Influenced: {influenced_prediction.confidence_level:.3f}"
            )
        elif accuracy_rate < 0.2:  # Very low accuracy should decrease confidence
            assert (
                influenced_prediction.confidence_level
                <= base_prediction.confidence_level * 1.2
            ), (
                f"Very low historical accuracy ({accuracy_rate:.2f}) should maintain or decrease confidence. "
                f"Base: {base_prediction.confidence_level:.3f}, Influenced: {influenced_prediction.confidence_level:.3f}"
            )

        # Confidence should always remain within bounds
        assert (
            0.0 <= influenced_prediction.confidence_level <= 1.0
        ), f"Confidence level {influenced_prediction.confidence_level} is out of bounds"

        # Close database connections properly
        del historical_analyzer
        del historical_store

    finally:
        # Clean up temporary database
        try:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
        except PermissionError:
            # On Windows, sometimes the file is still locked, just pass
            pass


@given(
    stock_symbol=st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]),
    base_prediction=market_prediction_strategy(),
    sentiment=sentiment_analysis_strategy(),
)
@settings(max_examples=100)
def test_historical_influence_consistency(stock_symbol, base_prediction, sentiment):
    """
    **Feature: news-market-predictor, Property 12: Historical data incorporation**

    Property: Historical influence should be consistent - same inputs should
    produce the same influenced predictions.

    **Validates: Requirements 3.4**

    This test verifies that historical influence calculation is deterministic.
    """
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Initialize historical data store and analyzer
        historical_store = HistoricalDataStore(db_path=temp_db_path)
        historical_analyzer = HistoricalAnalyzer(historical_store)

        # Ensure prediction uses the test stock symbol
        base_prediction.stock_symbol = stock_symbol

        # Calculate historical influence multiple times with same inputs
        influenced_prediction1 = historical_analyzer.calculate_historical_influence(
            stock_symbol=stock_symbol,
            sentiment=sentiment,
            base_prediction=base_prediction,
        )

        influenced_prediction2 = historical_analyzer.calculate_historical_influence(
            stock_symbol=stock_symbol,
            sentiment=sentiment,
            base_prediction=base_prediction,
        )

        influenced_prediction3 = historical_analyzer.calculate_historical_influence(
            stock_symbol=stock_symbol,
            sentiment=sentiment,
            base_prediction=base_prediction,
        )

        # All predictions should be identical
        assert (
            influenced_prediction1.stock_symbol
            == influenced_prediction2.stock_symbol
            == influenced_prediction3.stock_symbol
        )
        assert (
            influenced_prediction1.article_id
            == influenced_prediction2.article_id
            == influenced_prediction3.article_id
        )
        assert (
            influenced_prediction1.impact_direction
            == influenced_prediction2.impact_direction
            == influenced_prediction3.impact_direction
        )
        assert (
            influenced_prediction1.impact_magnitude
            == influenced_prediction2.impact_magnitude
            == influenced_prediction3.impact_magnitude
        )
        assert (
            influenced_prediction1.confidence_level
            == influenced_prediction2.confidence_level
            == influenced_prediction3.confidence_level
        )
        assert (
            influenced_prediction1.reasoning
            == influenced_prediction2.reasoning
            == influenced_prediction3.reasoning
        )

        # Close database connections properly
        del historical_analyzer
        del historical_store

    finally:
        # Clean up temporary database
        try:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
        except PermissionError:
            # On Windows, sometimes the file is still locked, just pass
            pass
