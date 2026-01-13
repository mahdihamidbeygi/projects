"""
Property-based tests for accuracy calculation correctness.

**Feature: news-market-predictor, Property 15: Accuracy calculation correctness**
"""

import os
import tempfile
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, HealthCheck

from news_market_predictor.storage.historical_data_store import HistoricalDataStore
from news_market_predictor.models import (
    MarketPrediction,
    MarketOutcome,
    HistoricalAccuracy,
)


@given(
    stock_symbol=st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]),
    correct_ratio=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    total_predictions=st.integers(min_value=5, max_value=20),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_accuracy_calculation_with_controlled_correctness(
    stock_symbol, correct_ratio, total_predictions
):
    """
    **Feature: news-market-predictor, Property 15: Accuracy calculation correctness**

    Property: For any controlled set of predictions with known correctness ratio,
    the calculated accuracy should match the expected ratio.

    **Validates: Requirements 4.3**

    This test verifies accuracy calculation with precisely controlled data.
    """
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Initialize historical data store
        historical_store = HistoricalDataStore(db_path=temp_db_path)

        # Calculate how many predictions should be correct
        correct_predictions_count = int(total_predictions * correct_ratio)

        # Use recent dates that will be within the 30-day window from now
        base_date = datetime.now() - timedelta(days=15)  # 15 days ago

        # Create predictions and outcomes with controlled correctness
        for i in range(total_predictions):
            # Create prediction
            prediction = MarketPrediction(
                article_id=f"article_{i}",
                stock_symbol=stock_symbol,
                impact_direction=(
                    "positive" if i < correct_predictions_count else "negative"
                ),
                impact_magnitude=0.5,
                confidence_level=0.7,
                reasoning=f"Test prediction {i}",
                created_at=base_date - timedelta(days=i % 15),  # Spread over 15 days
            )

            # Create matching or non-matching outcome
            outcome = MarketOutcome(
                prediction_id=f"article_{i}",
                stock_symbol=stock_symbol,
                actual_direction="positive",  # All outcomes are positive
                actual_magnitude=0.5,
                price_change_percent=2.5,
                evaluation_date=base_date - timedelta(days=i % 15),
                time_horizon_hours=24,
            )

            historical_store.store_prediction(prediction)
            historical_store.store_outcome(outcome)

        # Calculate historical accuracy
        calculated_accuracy = historical_store.calculate_historical_accuracy(
            stock_symbol=stock_symbol, time_period_days=30
        )

        # Verify accuracy matches expected ratio
        assert calculated_accuracy is not None
        assert calculated_accuracy.total_predictions == total_predictions
        assert calculated_accuracy.correct_predictions == correct_predictions_count

        expected_accuracy_rate = correct_predictions_count / total_predictions
        assert (
            abs(calculated_accuracy.accuracy_rate - expected_accuracy_rate) < 0.001
        ), (
            f"Calculated accuracy rate {calculated_accuracy.accuracy_rate:.3f} "
            f"should match expected rate {expected_accuracy_rate:.3f}. "
            f"Correct: {correct_predictions_count}, Total: {total_predictions}"
        )

        # Verify accuracy rate is within valid bounds
        assert (
            0.0 <= calculated_accuracy.accuracy_rate <= 1.0
        ), f"Accuracy rate {calculated_accuracy.accuracy_rate} must be between 0.0 and 1.0"

        # Close database connections properly
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
)
@settings(max_examples=50)
def test_accuracy_calculation_with_no_predictions(stock_symbol):
    """
    **Feature: news-market-predictor, Property 15: Accuracy calculation correctness**

    Property: When no predictions exist for a stock, accuracy calculation should
    return None to indicate insufficient data.

    **Validates: Requirements 4.3**

    This test verifies that accuracy calculation handles empty datasets correctly.
    """
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Initialize historical data store (empty database)
        historical_store = HistoricalDataStore(db_path=temp_db_path)

        # Calculate historical accuracy with no data
        calculated_accuracy = historical_store.calculate_historical_accuracy(
            stock_symbol=stock_symbol, time_period_days=30
        )

        # Should return None when no predictions exist
        assert (
            calculated_accuracy is None
        ), "Accuracy calculation should return None when no predictions exist"

        # Close database connections properly
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
    num_predictions=st.integers(min_value=5, max_value=15),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_accuracy_calculation_with_no_outcomes(stock_symbol, num_predictions):
    """
    **Feature: news-market-predictor, Property 15: Accuracy calculation correctness**

    Property: When predictions exist but no outcomes are available, accuracy
    calculation should handle this gracefully and return appropriate values.

    **Validates: Requirements 4.3**

    This test verifies that accuracy calculation handles missing outcome data correctly.
    """
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Initialize historical data store
        historical_store = HistoricalDataStore(db_path=temp_db_path)

        # Use recent dates that will be within the 30-day window from now
        base_date = datetime.now() - timedelta(days=15)

        # Store predictions without outcomes
        for i in range(num_predictions):
            prediction = MarketPrediction(
                article_id=f"article_{i}",
                stock_symbol=stock_symbol,
                impact_direction="positive",
                impact_magnitude=0.5,
                confidence_level=0.7,
                reasoning=f"Test prediction {i}",
                created_at=base_date - timedelta(days=i % 15),
            )
            historical_store.store_prediction(prediction)

        # Calculate historical accuracy
        calculated_accuracy = historical_store.calculate_historical_accuracy(
            stock_symbol=stock_symbol, time_period_days=30
        )

        # When no outcomes exist, accuracy should be 0.0 (no correct predictions)
        assert (
            calculated_accuracy is not None
        ), "Should return accuracy object even with no outcomes"
        assert calculated_accuracy.stock_symbol == stock_symbol
        assert calculated_accuracy.total_predictions == num_predictions
        assert calculated_accuracy.correct_predictions == 0
        assert (
            calculated_accuracy.accuracy_rate == 0.0
        ), "Accuracy rate should be 0.0 when no outcomes are available"

        # Average confidence should still be calculated from predictions
        expected_avg_confidence = 0.7  # All predictions have 0.7 confidence
        assert (
            abs(calculated_accuracy.average_confidence - expected_avg_confidence)
            < 0.001
        ), (
            f"Average confidence {calculated_accuracy.average_confidence:.3f} "
            f"should match expected {expected_avg_confidence:.3f}"
        )

        # Close database connections properly
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
    time_period_days=st.integers(min_value=1, max_value=60),
    num_predictions=st.integers(min_value=6, max_value=20),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_accuracy_calculation_respects_time_period(
    stock_symbol, time_period_days, num_predictions
):
    """
    **Feature: news-market-predictor, Property 15: Accuracy calculation correctness**

    Property: Accuracy calculation should only include predictions within the
    specified time period, regardless of other data in the database.

    **Validates: Requirements 4.3**

    This test verifies that time period filtering works correctly in accuracy calculation.
    """
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Initialize historical data store
        historical_store = HistoricalDataStore(db_path=temp_db_path)

        # Create predictions both inside and outside the time period
        cutoff_date = datetime.now() - timedelta(days=time_period_days)

        predictions_in_period = 0
        correct_in_period = 0

        for i in range(num_predictions):
            # Half the predictions are within the time period, half are older
            if i % 2 == 0:
                # Within time period
                created_at = cutoff_date + timedelta(days=1)
                predictions_in_period += 1
                # Make half of in-period predictions correct
                impact_direction = "positive" if i % 4 == 0 else "negative"
                if impact_direction == "positive":
                    correct_in_period += 1
            else:
                # Outside time period (older)
                created_at = cutoff_date - timedelta(days=1)
                impact_direction = "positive"

            # Create prediction
            prediction = MarketPrediction(
                article_id=f"article_{i}",
                stock_symbol=stock_symbol,
                impact_direction=impact_direction,
                impact_magnitude=0.5,
                confidence_level=0.7,
                reasoning=f"Test prediction {i}",
                created_at=created_at,
            )

            # Create outcome (all positive)
            outcome = MarketOutcome(
                prediction_id=f"article_{i}",
                stock_symbol=stock_symbol,
                actual_direction="positive",
                actual_magnitude=0.5,
                price_change_percent=2.5,
                evaluation_date=created_at,
                time_horizon_hours=24,
            )

            historical_store.store_prediction(prediction)
            historical_store.store_outcome(outcome)

        # Calculate accuracy for the specified time period
        calculated_accuracy = historical_store.calculate_historical_accuracy(
            stock_symbol=stock_symbol, time_period_days=time_period_days
        )

        # Verify only predictions within time period are counted
        if predictions_in_period > 0:
            assert calculated_accuracy is not None
            assert calculated_accuracy.total_predictions == predictions_in_period, (
                f"Should only count {predictions_in_period} predictions within time period, "
                f"but got {calculated_accuracy.total_predictions}"
            )
            assert calculated_accuracy.correct_predictions == correct_in_period

            expected_accuracy = correct_in_period / predictions_in_period
            assert (
                abs(calculated_accuracy.accuracy_rate - expected_accuracy) < 0.001
            ), f"Accuracy rate should be {expected_accuracy:.3f}, got {calculated_accuracy.accuracy_rate:.3f}"
        else:
            # If no predictions in period, should return None
            assert calculated_accuracy is None

        # Close database connections properly
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
    num_predictions=st.integers(min_value=5, max_value=15),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_accuracy_calculation_bounds_validation(stock_symbol, num_predictions):
    """
    **Feature: news-market-predictor, Property 15: Accuracy calculation correctness**

    Property: For any accuracy calculation, the accuracy rate should always be
    between 0.0 and 1.0, and all fields should be valid.

    **Validates: Requirements 4.3**

    This test verifies that accuracy calculation always produces valid results.
    """
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Initialize historical data store
        historical_store = HistoricalDataStore(db_path=temp_db_path)

        # Use recent dates that will be within the 30-day window from now
        base_date = datetime.now() - timedelta(days=15)

        # Create predictions and outcomes with random correctness
        for i in range(num_predictions):
            # Randomly make some predictions correct
            impact_direction = "positive" if i % 3 == 0 else "negative"

            prediction = MarketPrediction(
                article_id=f"article_{i}",
                stock_symbol=stock_symbol,
                impact_direction=impact_direction,
                impact_magnitude=0.5,
                confidence_level=0.6 + (i % 4) * 0.1,  # Vary confidence
                reasoning=f"Test prediction {i}",
                created_at=base_date - timedelta(days=i % 15),
            )

            # Create outcome (all positive, so only positive predictions are correct)
            outcome = MarketOutcome(
                prediction_id=f"article_{i}",
                stock_symbol=stock_symbol,
                actual_direction="positive",
                actual_magnitude=0.5,
                price_change_percent=2.5,
                evaluation_date=base_date - timedelta(days=i % 15),
                time_horizon_hours=24,
            )

            historical_store.store_prediction(prediction)
            historical_store.store_outcome(outcome)

        # Calculate historical accuracy
        calculated_accuracy = historical_store.calculate_historical_accuracy(
            stock_symbol=stock_symbol, time_period_days=30
        )

        # Verify accuracy calculation produces valid results
        assert calculated_accuracy is not None
        assert calculated_accuracy.stock_symbol == stock_symbol
        assert calculated_accuracy.time_period_days == 30
        assert calculated_accuracy.total_predictions == num_predictions
        assert 0 <= calculated_accuracy.correct_predictions <= num_predictions

        # The key property: accuracy rate should always be between 0.0 and 1.0
        assert (
            0.0 <= calculated_accuracy.accuracy_rate <= 1.0
        ), f"Accuracy rate {calculated_accuracy.accuracy_rate} must be between 0.0 and 1.0"

        # Average confidence should be between 0.0 and 1.0
        assert (
            0.0 <= calculated_accuracy.average_confidence <= 1.0
        ), f"Average confidence {calculated_accuracy.average_confidence} must be between 0.0 and 1.0"

        # Accuracy rate should match the ratio of correct to total predictions
        expected_accuracy_rate = (
            calculated_accuracy.correct_predictions
            / calculated_accuracy.total_predictions
        )
        assert (
            abs(calculated_accuracy.accuracy_rate - expected_accuracy_rate) < 0.001
        ), (
            f"Accuracy rate {calculated_accuracy.accuracy_rate:.3f} should match "
            f"correct/total ratio {expected_accuracy_rate:.3f}"
        )

        # Close database connections properly
        del historical_store

    finally:
        # Clean up temporary database
        try:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
        except PermissionError:
            # On Windows, sometimes the file is still locked, just pass
            pass
