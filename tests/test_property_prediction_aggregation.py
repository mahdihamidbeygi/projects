"""
Property-based tests for prediction aggregation consistency.

**Feature: news-market-predictor, Property 16: Prediction aggregation consistency**
"""

from datetime import datetime
from hypothesis import given, strategies as st, settings
from statistics import mean

from news_market_predictor.aggregator.results_aggregator import ResultsAggregatorImpl
from news_market_predictor.models import MarketPrediction


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


@given(
    stock_symbol=st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]),
    predictions=st.lists(market_prediction_strategy(), min_size=2, max_size=10),
)
@settings(max_examples=100)
def test_prediction_aggregation_consistency_weighted_confidence_scores(
    stock_symbol, predictions
):
    """
    **Feature: news-market-predictor, Property 16: Prediction aggregation consistency**

    Property: For any stock with multiple predictions, the aggregated result should
    properly weight individual predictions by their confidence scores.

    **Validates: Requirements 4.4**

    This test verifies that when multiple predictions exist for the same stock,
    they are aggregated using weighted confidence scores as specified in requirement 4.4.
    """
    # Ensure all predictions are for the same stock symbol
    for prediction in predictions:
        prediction.stock_symbol = stock_symbol

    # Initialize the results aggregator
    aggregator = ResultsAggregatorImpl()

    # Aggregate the predictions
    aggregated_predictions = aggregator.aggregate_predictions(predictions)

    # Should return exactly one aggregated prediction for the stock
    assert len(aggregated_predictions) == 1, "Should aggregate to single prediction"
    aggregated = aggregated_predictions[0]

    # Verify the aggregated prediction is for the correct stock
    assert aggregated.stock_symbol == stock_symbol, "Stock symbol should be preserved"

    # Calculate expected weighted averages manually to verify correctness
    total_weight = sum(p.confidence_level for p in predictions)

    if total_weight > 0:
        # When there are non-zero confidence levels, use weighted averaging
        expected_weighted_magnitude = (
            sum(p.impact_magnitude * p.confidence_level for p in predictions)
            / total_weight
        )

        expected_weighted_confidence = (
            sum(p.confidence_level * p.confidence_level for p in predictions)
            / total_weight
        )

        # Verify the aggregated values match expected weighted averages
        assert abs(aggregated.impact_magnitude - expected_weighted_magnitude) < 0.001, (
            f"Impact magnitude should be weighted average: "
            f"expected {expected_weighted_magnitude}, got {aggregated.impact_magnitude}"
        )

        assert (
            abs(aggregated.confidence_level - expected_weighted_confidence) < 0.001
        ), (
            f"Confidence level should be weighted average: "
            f"expected {expected_weighted_confidence}, got {aggregated.confidence_level}"
        )
    else:
        # When all confidence levels are zero, should use simple averaging
        expected_magnitude = mean(p.impact_magnitude for p in predictions)
        expected_confidence = mean(p.confidence_level for p in predictions)

        assert abs(aggregated.impact_magnitude - expected_magnitude) < 0.001, (
            f"Impact magnitude should be simple average when all confidence is zero: "
            f"expected {expected_magnitude}, got {aggregated.impact_magnitude}"
        )

        assert abs(aggregated.confidence_level - expected_confidence) < 0.001, (
            f"Confidence level should be simple average when all confidence is zero: "
            f"expected {expected_confidence}, got {aggregated.confidence_level}"
        )

    # Verify the aggregated direction is determined by weighted impact scores
    direction_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    if total_weight > 0:
        weights = [p.confidence_level / total_weight for p in predictions]
    else:
        weights = [1.0 / len(predictions)] * len(predictions)

    for prediction, weight in zip(predictions, weights):
        if prediction.impact_direction == "positive":
            direction_scores["positive"] += weight * prediction.impact_magnitude
        elif prediction.impact_direction == "negative":
            direction_scores["negative"] += weight * prediction.impact_magnitude
        else:
            direction_scores["neutral"] += weight

    expected_direction = max(direction_scores.items(), key=lambda x: x[1])[0]
    assert aggregated.impact_direction == expected_direction, (
        f"Direction should be weighted by confidence: "
        f"expected {expected_direction}, got {aggregated.impact_direction}"
    )

    # Verify reasoning combines all individual predictions
    assert (
        "Aggregated from" in aggregated.reasoning
    ), "Reasoning should indicate aggregation"
    assert (
        str(len(predictions)) in aggregated.reasoning
    ), "Reasoning should mention number of predictions"

    # Verify article_id indicates aggregation
    assert aggregated.article_id.startswith(
        "AGG_"
    ), "Aggregated prediction should have AGG_ prefix in article_id"


@given(
    predictions_by_stock=st.dictionaries(
        keys=st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA"]),
        values=st.lists(market_prediction_strategy(), min_size=2, max_size=5),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=100)
def test_prediction_aggregation_consistency_multiple_stocks(predictions_by_stock):
    """
    **Feature: news-market-predictor, Property 16: Prediction aggregation consistency**

    Property: For any set of predictions grouped by stock, each stock's predictions
    should be aggregated independently with proper confidence weighting.

    **Validates: Requirements 4.4**

    This test verifies that aggregation works correctly when there are multiple
    stocks, each with multiple predictions.
    """
    # Flatten predictions and ensure stock symbols match the grouping
    all_predictions = []
    for stock_symbol, stock_predictions in predictions_by_stock.items():
        for prediction in stock_predictions:
            prediction.stock_symbol = stock_symbol
            all_predictions.append(prediction)

    # Initialize the results aggregator
    aggregator = ResultsAggregatorImpl()

    # Aggregate all predictions
    aggregated_predictions = aggregator.aggregate_predictions(all_predictions)

    # Should have one aggregated prediction per stock
    assert len(aggregated_predictions) == len(predictions_by_stock), (
        f"Should have one prediction per stock: "
        f"expected {len(predictions_by_stock)}, got {len(aggregated_predictions)}"
    )

    # Verify each stock has exactly one aggregated prediction
    aggregated_stocks = {pred.stock_symbol for pred in aggregated_predictions}
    expected_stocks = set(predictions_by_stock.keys())
    assert aggregated_stocks == expected_stocks, (
        f"Aggregated stocks should match input stocks: "
        f"expected {expected_stocks}, got {aggregated_stocks}"
    )

    # Verify each aggregated prediction properly weights its stock's predictions
    for aggregated in aggregated_predictions:
        stock_symbol = aggregated.stock_symbol
        original_predictions = predictions_by_stock[stock_symbol]

        # Calculate expected weighted values for this stock
        total_weight = sum(p.confidence_level for p in original_predictions)

        if total_weight > 0:
            expected_magnitude = (
                sum(
                    p.impact_magnitude * p.confidence_level
                    for p in original_predictions
                )
                / total_weight
            )

            expected_confidence = (
                sum(
                    p.confidence_level * p.confidence_level
                    for p in original_predictions
                )
                / total_weight
            )
        else:
            expected_magnitude = mean(p.impact_magnitude for p in original_predictions)
            expected_confidence = mean(p.confidence_level for p in original_predictions)

        # Verify weighted aggregation for this stock
        assert abs(aggregated.impact_magnitude - expected_magnitude) < 0.001, (
            f"Stock {stock_symbol} magnitude should be weighted average: "
            f"expected {expected_magnitude}, got {aggregated.impact_magnitude}"
        )

        assert abs(aggregated.confidence_level - expected_confidence) < 0.001, (
            f"Stock {stock_symbol} confidence should be weighted average: "
            f"expected {expected_confidence}, got {aggregated.confidence_level}"
        )


@given(
    stock_symbol=st.sampled_from(["AAPL", "GOOGL", "MSFT"]),
    predictions=st.lists(market_prediction_strategy(), min_size=2, max_size=8),
)
@settings(max_examples=100)
def test_prediction_aggregation_consistency_confidence_weighting_effect(
    stock_symbol, predictions
):
    """
    **Feature: news-market-predictor, Property 16: Prediction aggregation consistency**

    Property: For any stock with multiple predictions, higher confidence predictions
    should have more influence on the aggregated result than lower confidence ones.

    **Validates: Requirements 4.4**

    This test verifies that the confidence weighting actually affects the outcome,
    ensuring high-confidence predictions dominate the aggregation.
    """
    # Ensure all predictions are for the same stock
    for prediction in predictions:
        prediction.stock_symbol = stock_symbol

    # Create a modified version where we boost one prediction's confidence significantly
    modified_predictions = predictions.copy()
    if len(modified_predictions) >= 2:
        # Set the first prediction to have very high confidence
        modified_predictions[0].confidence_level = 0.95
        modified_predictions[0].impact_magnitude = 0.8
        modified_predictions[0].impact_direction = "positive"

        # Set others to have low confidence
        for i in range(1, len(modified_predictions)):
            modified_predictions[i].confidence_level = 0.05
            modified_predictions[i].impact_magnitude = 0.2
            modified_predictions[i].impact_direction = "negative"

        # Initialize the results aggregator
        aggregator = ResultsAggregatorImpl()

        # Aggregate the modified predictions
        aggregated_predictions = aggregator.aggregate_predictions(modified_predictions)
        aggregated = aggregated_predictions[0]

        # The high-confidence prediction should dominate the result
        # Since the high-confidence prediction has 0.95 confidence vs others with 0.05,
        # it should heavily influence the outcome

        total_weight = sum(p.confidence_level for p in modified_predictions)
        high_conf_weight = 0.95 / total_weight

        # The aggregated magnitude should be closer to the high-confidence prediction
        # than to a simple average
        simple_average_magnitude = mean(
            p.impact_magnitude for p in modified_predictions
        )

        # High confidence prediction should pull the result toward its value
        assert aggregated.impact_magnitude > simple_average_magnitude, (
            f"High confidence prediction should pull magnitude up: "
            f"aggregated {aggregated.impact_magnitude} should be > "
            f"simple average {simple_average_magnitude}"
        )

        # If the high-confidence prediction has enough weight, it should determine direction
        if high_conf_weight > 0.5:  # More than half the weight
            assert aggregated.impact_direction == "positive", (
                f"High confidence prediction should determine direction when dominant: "
                f"expected 'positive', got '{aggregated.impact_direction}'"
            )


@given(
    single_prediction=market_prediction_strategy(),
)
@settings(max_examples=100)
def test_prediction_aggregation_consistency_single_prediction_unchanged(
    single_prediction,
):
    """
    **Feature: news-market-predictor, Property 16: Prediction aggregation consistency**

    Property: For any stock with only one prediction, the aggregation should return
    the original prediction unchanged.

    **Validates: Requirements 4.4**

    This test verifies that single predictions are not modified during aggregation,
    which is an important edge case of the aggregation consistency requirement.
    """
    # Initialize the results aggregator
    aggregator = ResultsAggregatorImpl()

    # Aggregate a single prediction
    aggregated_predictions = aggregator.aggregate_predictions([single_prediction])

    # Should return exactly one prediction
    assert len(aggregated_predictions) == 1, "Should return single prediction"
    result = aggregated_predictions[0]

    # The result should be identical to the original (no aggregation needed)
    assert result.stock_symbol == single_prediction.stock_symbol
    assert result.impact_direction == single_prediction.impact_direction
    assert result.impact_magnitude == single_prediction.impact_magnitude
    assert result.confidence_level == single_prediction.confidence_level
    assert result.reasoning == single_prediction.reasoning
    assert result.article_id == single_prediction.article_id
    assert result.created_at == single_prediction.created_at


@given(
    predictions=st.lists(market_prediction_strategy(), min_size=2, max_size=6),
)
@settings(max_examples=100)
def test_prediction_aggregation_consistency_zero_confidence_handling(predictions):
    """
    **Feature: news-market-predictor, Property 16: Prediction aggregation consistency**

    Property: For any stock with multiple predictions where all have zero confidence,
    the aggregation should use simple averaging instead of weighted averaging.

    **Validates: Requirements 4.4**

    This test verifies the edge case where confidence-based weighting cannot be used.
    """
    # Set all predictions to the same stock and zero confidence
    stock_symbol = "AAPL"
    for prediction in predictions:
        prediction.stock_symbol = stock_symbol
        prediction.confidence_level = 0.0

    # Initialize the results aggregator
    aggregator = ResultsAggregatorImpl()

    # Aggregate the zero-confidence predictions
    aggregated_predictions = aggregator.aggregate_predictions(predictions)
    aggregated = aggregated_predictions[0]

    # Should use simple averaging when all confidence levels are zero
    expected_magnitude = mean(p.impact_magnitude for p in predictions)
    expected_confidence = mean(p.confidence_level for p in predictions)  # Should be 0.0

    assert abs(aggregated.impact_magnitude - expected_magnitude) < 0.001, (
        f"Should use simple average for magnitude when confidence is zero: "
        f"expected {expected_magnitude}, got {aggregated.impact_magnitude}"
    )

    assert abs(aggregated.confidence_level - expected_confidence) < 0.001, (
        f"Should use simple average for confidence when all are zero: "
        f"expected {expected_confidence}, got {aggregated.confidence_level}"
    )

    # Direction should still be determined by the averaging logic
    direction_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for prediction in predictions:
        direction_counts[prediction.impact_direction] += 1

    # With equal weights (1/n for each), the direction with most predictions
    # or highest combined magnitude should win
    assert aggregated.impact_direction in ["positive", "negative", "neutral"]
