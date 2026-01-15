"""
Property-based tests for storage failure recovery.

**Feature: news-market-predictor, Property 21: Storage failure recovery**
"""

import logging
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Optional

from hypothesis import given, strategies as st, settings, assume

from news_market_predictor.storage.historical_data_store import HistoricalDataStore
from news_market_predictor.error_handling import StorageFailureRecovery
from news_market_predictor.models import (
    MarketPrediction,
    MarketOutcome,
    HistoricalAccuracy,
)
from news_market_predictor.exceptions import StorageError


# Strategies for generating test data
@st.composite
def prediction_strategy(draw):
    """Generate valid MarketPrediction objects for testing."""
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    return MarketPrediction(
        article_id=draw(st.text(min_size=1, max_size=50)),
        stock_symbol=draw(st.sampled_from(stock_symbols)),
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
        reasoning=draw(st.text(min_size=1, max_size=200)),
        created_at=draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
            )
        ),
    )


@st.composite
def outcome_strategy(draw):
    """Generate valid MarketOutcome objects for testing."""
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    return MarketOutcome(
        prediction_id=draw(st.text(min_size=1, max_size=50)),
        stock_symbol=draw(st.sampled_from(stock_symbols)),
        actual_direction=draw(st.sampled_from(["positive", "negative", "neutral"])),
        actual_magnitude=draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        price_change_percent=draw(
            st.floats(
                min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False
            )
        ),
        evaluation_date=draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
            )
        ),
        time_horizon_hours=draw(st.integers(min_value=1, max_value=168)),
    )


@st.composite
def accuracy_strategy(draw):
    """Generate valid HistoricalAccuracy objects for testing."""
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    total_predictions = draw(st.integers(min_value=1, max_value=100))
    correct_predictions = draw(st.integers(min_value=0, max_value=total_predictions))

    return HistoricalAccuracy(
        stock_symbol=draw(st.sampled_from(stock_symbols)),
        time_period_days=draw(st.integers(min_value=1, max_value=365)),
        total_predictions=total_predictions,
        correct_predictions=correct_predictions,
        accuracy_rate=(
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        ),
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


@given(prediction=prediction_strategy())
@settings(max_examples=100, deadline=5000)
def test_storage_failure_attempts_backup_storage(prediction):
    """
    **Feature: news-market-predictor, Property 21: Storage failure recovery**

    Property: For any storage operation failure, the system should attempt
    backup storage and generate administrator alerts.

    **Validates: Requirements 5.4**

    This test verifies that when primary storage fails, the system attempts
    to use backup storage as specified in the requirements.
    """
    # Create mock storage objects
    primary_storage = Mock()
    backup_storage = Mock()

    # Make primary storage fail
    primary_storage.store_prediction.side_effect = StorageError(
        "Primary storage failed"
    )

    # Make backup storage succeed
    backup_storage.store_prediction.return_value = True

    # Create recovery handler
    recovery = StorageFailureRecovery(primary_storage, backup_storage)

    # Mock the alert method to track if it was called
    with patch.object(recovery, "_alert_administrators") as mock_alert:
        # Attempt storage operation
        result = recovery.store_with_recovery("store_prediction", prediction)

        # Should have succeeded using backup
        assert result is True, "Should succeed using backup storage"

        # Should have called backup storage
        backup_storage.store_prediction.assert_called_once_with(prediction)

        # Should have alerted administrators about primary failure
        assert (
            mock_alert.call_count >= 1
        ), "Should alert administrators when primary storage fails"


@given(prediction=prediction_strategy())
@settings(max_examples=100, deadline=5000)
def test_storage_failure_alerts_administrators_on_complete_failure(prediction):
    """
    **Feature: news-market-predictor, Property 21: Storage failure recovery**

    Property: For any storage operation where both primary and backup fail,
    the system should generate administrator alerts.

    **Validates: Requirements 5.4**

    This test verifies that when all storage systems fail, administrators
    are alerted as required.
    """
    # Create mock storage objects
    primary_storage = Mock()
    backup_storage = Mock()

    # Make both storages fail
    primary_storage.store_prediction.side_effect = StorageError(
        "Primary storage failed"
    )
    backup_storage.store_prediction.side_effect = StorageError("Backup storage failed")

    # Create recovery handler
    recovery = StorageFailureRecovery(primary_storage, backup_storage)

    # Mock the alert method to track calls
    with patch.object(recovery, "_alert_administrators") as mock_alert:
        # Attempt storage operation
        result = recovery.store_with_recovery("store_prediction", prediction)

        # Should have failed
        assert result is False, "Should return False when all storage systems fail"

        # Should have alerted administrators about complete failure
        assert (
            mock_alert.call_count >= 1
        ), "Should alert administrators when all storage fails"

        # Check that alert was called with appropriate message
        alert_calls = [str(call) for call in mock_alert.call_args_list]
        assert any(
            "failed" in str(call).lower() for call in alert_calls
        ), "Alert should mention storage failure"


@given(
    prediction=prediction_strategy(),
    failure_count=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50, deadline=5000)
def test_storage_failure_recovery_tracks_failure_count(prediction, failure_count):
    """
    **Feature: news-market-predictor, Property 21: Storage failure recovery**

    Property: For any sequence of storage failures, the system should track
    failure counts and mark primary storage as failed after multiple failures.

    **Validates: Requirements 5.4**

    This test verifies that the recovery system tracks failures and adjusts
    behavior accordingly.
    """
    # Create mock storage objects
    primary_storage = Mock()
    backup_storage = Mock()

    # Make primary storage fail
    primary_storage.store_prediction.side_effect = StorageError(
        "Primary storage failed"
    )

    # Make backup storage succeed
    backup_storage.store_prediction.return_value = True

    # Create recovery handler
    recovery = StorageFailureRecovery(primary_storage, backup_storage)

    # Mock the alert method
    with patch.object(recovery, "_alert_administrators"):
        # Attempt storage operations multiple times
        for i in range(failure_count):
            result = recovery.store_with_recovery("store_prediction", prediction)
            assert result is True, f"Attempt {i+1} should succeed using backup"

        # Check failure tracking
        # After 3 failures, primary is marked as failed and subsequent attempts
        # don't increment the failure count (they go directly to backup)
        expected_failure_count = min(failure_count, 3)
        assert (
            recovery.failure_count == expected_failure_count
        ), f"Should track {expected_failure_count} failures (capped at 3)"

        # After 3+ failures, primary should be marked as failed
        if failure_count >= 3:
            assert (
                recovery.primary_failed is True
            ), "Primary storage should be marked as failed after 3+ failures"
        else:
            assert (
                recovery.primary_failed is False
            ), "Primary storage should not be marked as failed before 3 failures"


@given(outcome=outcome_strategy())
@settings(max_examples=100, deadline=5000)
def test_storage_failure_recovery_works_for_different_operations(outcome):
    """
    **Feature: news-market-predictor, Property 21: Storage failure recovery**

    Property: For any storage operation type (prediction, outcome, accuracy),
    the failure recovery mechanism should work consistently.

    **Validates: Requirements 5.4**

    This test verifies that storage failure recovery works for all types
    of storage operations, not just predictions.
    """
    # Create mock storage objects
    primary_storage = Mock()
    backup_storage = Mock()

    # Make primary storage fail
    primary_storage.store_outcome.side_effect = StorageError("Primary storage failed")

    # Make backup storage succeed
    backup_storage.store_outcome.return_value = True

    # Create recovery handler
    recovery = StorageFailureRecovery(primary_storage, backup_storage)

    # Mock the alert method
    with patch.object(recovery, "_alert_administrators") as mock_alert:
        # Attempt storage operation for outcome
        result = recovery.store_with_recovery("store_outcome", outcome)

        # Should have succeeded using backup
        assert result is True, "Should succeed using backup storage for outcomes"

        # Should have called backup storage
        backup_storage.store_outcome.assert_called_once_with(outcome)

        # Should have alerted administrators
        assert mock_alert.call_count >= 1, "Should alert administrators"


@given(prediction=prediction_strategy())
@settings(max_examples=50, deadline=5000)
def test_storage_failure_recovery_without_backup(prediction):
    """
    **Feature: news-market-predictor, Property 21: Storage failure recovery**

    Property: For any storage failure when no backup is configured, the system
    should alert administrators and return failure status.

    **Validates: Requirements 5.4**

    This test verifies that the system handles the case where no backup
    storage is available.
    """
    # Create mock primary storage only (no backup)
    primary_storage = Mock()

    # Make primary storage fail
    primary_storage.store_prediction.side_effect = StorageError(
        "Primary storage failed"
    )

    # Create recovery handler without backup
    recovery = StorageFailureRecovery(primary_storage, backup_storage=None)

    # Mock the alert method
    with patch.object(recovery, "_alert_administrators") as mock_alert:
        # Attempt storage operation
        result = recovery.store_with_recovery("store_prediction", prediction)

        # Should have failed
        assert result is False, "Should return False when no backup is available"

        # Should have alerted administrators
        assert (
            mock_alert.call_count >= 1
        ), "Should alert administrators even without backup"


@given(prediction=prediction_strategy())
@settings(max_examples=100, deadline=5000)
def test_storage_success_resets_failure_state(prediction):
    """
    **Feature: news-market-predictor, Property 21: Storage failure recovery**

    Property: For any successful storage operation after failures, the system
    should reset failure tracking state.

    **Validates: Requirements 5.4**

    This test verifies that successful operations reset the failure state,
    allowing the system to recover from temporary issues.
    """
    # Create mock storage objects
    primary_storage = Mock()
    backup_storage = Mock()

    # Create recovery handler
    recovery = StorageFailureRecovery(primary_storage, backup_storage)

    # First, cause some failures
    primary_storage.store_prediction.side_effect = StorageError(
        "Primary storage failed"
    )
    backup_storage.store_prediction.return_value = True

    with patch.object(recovery, "_alert_administrators"):
        # Cause 2 failures
        recovery.store_with_recovery("store_prediction", prediction)
        recovery.store_with_recovery("store_prediction", prediction)

        # Should have 2 failures tracked
        assert recovery.failure_count == 2

    # Now make primary storage succeed
    primary_storage.store_prediction.side_effect = None
    primary_storage.store_prediction.return_value = True

    with patch.object(recovery, "_alert_administrators"):
        # Successful operation
        result = recovery.store_with_recovery("store_prediction", prediction)

        # Should succeed
        assert result is True

        # Failure state should be reset
        assert recovery.failure_count == 0, "Failure count should reset on success"
        assert (
            recovery.primary_failed is False
        ), "Primary failed flag should reset on success"


@given(
    prediction=prediction_strategy(),
    operation_name=st.sampled_from(
        ["store_prediction", "store_outcome", "store_accuracy_metrics"]
    ),
)
@settings(max_examples=50, deadline=5000)
def test_storage_failure_preserves_operation_parameters(prediction, operation_name):
    """
    **Feature: news-market-predictor, Property 21: Storage failure recovery**

    Property: For any storage operation that fails over to backup, the original
    operation parameters should be preserved and passed correctly to backup.

    **Validates: Requirements 5.4**

    This test verifies that failover doesn't corrupt or modify the data being stored.
    """
    # Create mock storage objects
    primary_storage = Mock()
    backup_storage = Mock()

    # Make primary storage fail
    setattr(primary_storage, operation_name, Mock(side_effect=StorageError("Failed")))

    # Make backup storage succeed and capture the call
    setattr(backup_storage, operation_name, Mock(return_value=True))

    # Create recovery handler
    recovery = StorageFailureRecovery(primary_storage, backup_storage)

    # Mock the alert method
    with patch.object(recovery, "_alert_administrators"):
        # Attempt storage operation
        result = recovery.store_with_recovery(operation_name, prediction)

        # Should succeed
        assert result is True

        # Verify backup was called with the same parameters
        backup_method = getattr(backup_storage, operation_name)
        backup_method.assert_called_once_with(prediction)

        # Verify the prediction object wasn't modified
        call_args = backup_method.call_args[0]
        assert (
            call_args[0] is prediction
        ), "Should pass the same object to backup storage"


@given(accuracy=accuracy_strategy())
@settings(max_examples=100, deadline=5000)
def test_storage_failure_recovery_for_accuracy_metrics(accuracy):
    """
    **Feature: news-market-predictor, Property 21: Storage failure recovery**

    Property: For any accuracy metrics storage failure, the system should
    attempt backup storage and alert administrators.

    **Validates: Requirements 5.4**

    This test verifies that storage failure recovery works specifically
    for accuracy metrics, which are critical for system monitoring.
    """
    # Create mock storage objects
    primary_storage = Mock()
    backup_storage = Mock()

    # Make primary storage fail
    primary_storage.store_accuracy_metrics.side_effect = StorageError(
        "Primary storage failed"
    )

    # Make backup storage succeed
    backup_storage.store_accuracy_metrics.return_value = True

    # Create recovery handler
    recovery = StorageFailureRecovery(primary_storage, backup_storage)

    # Mock the alert method
    with patch.object(recovery, "_alert_administrators") as mock_alert:
        # Attempt storage operation
        result = recovery.store_with_recovery("store_accuracy_metrics", accuracy)

        # Should have succeeded using backup
        assert (
            result is True
        ), "Should succeed using backup storage for accuracy metrics"

        # Should have called backup storage
        backup_storage.store_accuracy_metrics.assert_called_once_with(accuracy)

        # Should have alerted administrators
        assert (
            mock_alert.call_count >= 1
        ), "Should alert administrators about primary failure"


@given(prediction=prediction_strategy())
@settings(max_examples=50, deadline=5000)
def test_storage_failure_recovery_is_consistent(prediction):
    """
    **Feature: news-market-predictor, Property 21: Storage failure recovery**

    Property: For any storage failure scenario, the recovery behavior should
    be consistent across multiple identical failures.

    **Validates: Requirements 5.4**

    This test verifies that the recovery mechanism behaves deterministically
    and consistently for the same failure conditions.
    """
    # Create mock storage objects
    primary_storage = Mock()
    backup_storage = Mock()

    # Make primary storage fail consistently
    primary_storage.store_prediction.side_effect = StorageError(
        "Primary storage failed"
    )

    # Make backup storage succeed consistently
    backup_storage.store_prediction.return_value = True

    # Create recovery handler
    recovery = StorageFailureRecovery(primary_storage, backup_storage)

    # Mock the alert method
    with patch.object(recovery, "_alert_administrators"):
        # Attempt the same operation multiple times
        results = []
        for _ in range(3):
            result = recovery.store_with_recovery("store_prediction", prediction)
            results.append(result)

        # All results should be the same (all True, using backup)
        assert all(results), "All attempts should succeed consistently using backup"
        assert len(set(results)) == 1, "Recovery behavior should be consistent"


@given(
    prediction=prediction_strategy(),
    primary_fails=st.booleans(),
    backup_fails=st.booleans(),
)
@settings(max_examples=100, deadline=5000)
def test_storage_failure_recovery_handles_all_scenarios(
    prediction, primary_fails, backup_fails
):
    """
    **Feature: news-market-predictor, Property 21: Storage failure recovery**

    Property: For any combination of primary and backup storage states (success/failure),
    the system should handle the scenario appropriately and alert when necessary.

    **Validates: Requirements 5.4**

    This test verifies that all possible storage failure scenarios are handled correctly.
    """
    # Skip the case where both succeed and primary is not marked as failed
    # (this is the normal case and doesn't test recovery)
    assume(primary_fails or backup_fails)

    # Create mock storage objects
    primary_storage = Mock()
    backup_storage = Mock()

    # Configure primary storage
    if primary_fails:
        primary_storage.store_prediction.side_effect = StorageError("Primary failed")
    else:
        primary_storage.store_prediction.return_value = True

    # Configure backup storage
    if backup_fails:
        backup_storage.store_prediction.side_effect = StorageError("Backup failed")
    else:
        backup_storage.store_prediction.return_value = True

    # Create recovery handler
    recovery = StorageFailureRecovery(primary_storage, backup_storage)

    # Mock the alert method
    with patch.object(recovery, "_alert_administrators") as mock_alert:
        # Attempt storage operation
        result = recovery.store_with_recovery("store_prediction", prediction)

        # Verify expected behavior based on failure states
        if not primary_fails:
            # Primary succeeds - should return True
            assert result is True, "Should succeed when primary storage works"
            # Should not alert for successful operation
            assert mock_alert.call_count == 0, "Should not alert on success"
        elif primary_fails and not backup_fails:
            # Primary fails, backup succeeds - should return True
            assert result is True, "Should succeed using backup when primary fails"
            # Should alert about primary failure
            assert (
                mock_alert.call_count >= 1
            ), "Should alert when primary fails but backup succeeds"
        else:
            # Both fail - should return False
            assert result is False, "Should return False when all storage fails"
            # Should alert about complete failure
            assert mock_alert.call_count >= 1, "Should alert when all storage fails"
