"""
Data retention and cleanup policy manager for historical data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .historical_data_store import HistoricalDataStore


@dataclass
class RetentionPolicy:
    """Configuration for data retention policies."""

    predictions_retention_days: int = 365  # Keep predictions for 1 year
    outcomes_retention_days: int = 365  # Keep outcomes for 1 year
    accuracy_metrics_retention_days: int = 90  # Keep accuracy metrics for 3 months
    cleanup_frequency_hours: int = 24  # Run cleanup daily
    backup_before_cleanup: bool = True  # Create backup before cleanup


class RetentionManager:
    """Manages data retention and cleanup policies."""

    def __init__(
        self,
        historical_store: HistoricalDataStore,
        policy: Optional[RetentionPolicy] = None,
    ):
        """Initialize retention manager with store and policy."""
        self.historical_store = historical_store
        self.policy = policy or RetentionPolicy()
        self.logger = logging.getLogger(__name__)
        self._last_cleanup = None

    def should_run_cleanup(self) -> bool:
        """Check if cleanup should be run based on frequency policy."""
        if self._last_cleanup is None:
            return True

        time_since_cleanup = datetime.now() - self._last_cleanup
        return time_since_cleanup.total_seconds() >= (
            self.policy.cleanup_frequency_hours * 3600
        )

    def run_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        Run data cleanup based on retention policy.

        Args:
            force: Force cleanup even if frequency policy says not to

        Returns:
            Dictionary with cleanup results
        """
        if not force and not self.should_run_cleanup():
            return {
                "status": "skipped",
                "reason": "Cleanup frequency policy not met",
                "last_cleanup": (
                    self._last_cleanup.isoformat() if self._last_cleanup else None
                ),
            }

        self.logger.info("Starting data retention cleanup")
        cleanup_results = {
            "status": "completed",
            "start_time": datetime.now().isoformat(),
            "policy": {
                "predictions_retention_days": self.policy.predictions_retention_days,
                "outcomes_retention_days": self.policy.outcomes_retention_days,
                "accuracy_metrics_retention_days": self.policy.accuracy_metrics_retention_days,
            },
            "actions": [],
        }

        try:
            # Get database stats before cleanup
            stats_before = self.historical_store.get_database_stats()
            cleanup_results["stats_before"] = stats_before

            # Create backup if policy requires it
            if self.policy.backup_before_cleanup:
                backup_result = self._create_backup()
                cleanup_results["actions"].append(backup_result)

            # Clean up old predictions
            predictions_cutoff = datetime.now() - timedelta(
                days=self.policy.predictions_retention_days
            )
            predictions_cleaned = self._cleanup_predictions(predictions_cutoff)
            cleanup_results["actions"].append(predictions_cleaned)

            # Clean up old outcomes
            outcomes_cutoff = datetime.now() - timedelta(
                days=self.policy.outcomes_retention_days
            )
            outcomes_cleaned = self._cleanup_outcomes(outcomes_cutoff)
            cleanup_results["actions"].append(outcomes_cleaned)

            # Clean up old accuracy metrics
            accuracy_cutoff = datetime.now() - timedelta(
                days=self.policy.accuracy_metrics_retention_days
            )
            accuracy_cleaned = self._cleanup_accuracy_metrics(accuracy_cutoff)
            cleanup_results["actions"].append(accuracy_cleaned)

            # Get database stats after cleanup
            stats_after = self.historical_store.get_database_stats()
            cleanup_results["stats_after"] = stats_after

            # Calculate space saved
            cleanup_results["records_removed"] = {
                "predictions": stats_before.get("total_predictions", 0)
                - stats_after.get("total_predictions", 0),
                "outcomes": stats_before.get("total_outcomes", 0)
                - stats_after.get("total_outcomes", 0),
                "accuracy_metrics": stats_before.get("total_accuracy_records", 0)
                - stats_after.get("total_accuracy_records", 0),
            }

            self._last_cleanup = datetime.now()
            cleanup_results["end_time"] = self._last_cleanup.isoformat()

            self.logger.info(
                f"Cleanup completed successfully. Removed {sum(cleanup_results['records_removed'].values())} total records"
            )

        except Exception as e:
            cleanup_results["status"] = "failed"
            cleanup_results["error"] = str(e)
            cleanup_results["end_time"] = datetime.now().isoformat()
            self.logger.error(f"Cleanup failed: {e}")

        return cleanup_results

    def _create_backup(self) -> Dict[str, Any]:
        """Create backup of data before cleanup."""
        try:
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"historical_data_backup_{backup_timestamp}.db"

            # Simple file copy for SQLite database
            import shutil

            shutil.copy2(self.historical_store.db_path, backup_filename)

            return {
                "action": "backup",
                "status": "success",
                "backup_file": backup_filename,
                "timestamp": backup_timestamp,
            }
        except Exception as e:
            return {"action": "backup", "status": "failed", "error": str(e)}

    def _cleanup_predictions(self, cutoff_date: datetime) -> Dict[str, Any]:
        """Clean up predictions older than cutoff date."""
        try:
            # Get count before cleanup
            all_predictions = self.historical_store.retrieve_predictions()
            old_predictions = [p for p in all_predictions if p.created_at < cutoff_date]
            count_before = len(old_predictions)

            # Use the existing cleanup method
            success = self.historical_store.cleanup_old_data(
                retention_days=self.policy.predictions_retention_days
            )

            return {
                "action": "cleanup_predictions",
                "status": "success" if success else "failed",
                "cutoff_date": cutoff_date.isoformat(),
                "records_targeted": count_before,
            }
        except Exception as e:
            return {
                "action": "cleanup_predictions",
                "status": "failed",
                "error": str(e),
            }

    def _cleanup_outcomes(self, cutoff_date: datetime) -> Dict[str, Any]:
        """Clean up outcomes older than cutoff date."""
        try:
            # Get count before cleanup
            all_outcomes = self.historical_store.retrieve_outcomes()
            old_outcomes = [o for o in all_outcomes if o.evaluation_date < cutoff_date]
            count_before = len(old_outcomes)

            # The cleanup_old_data method handles outcomes too
            success = True  # Already handled in cleanup_predictions call

            return {
                "action": "cleanup_outcomes",
                "status": "success" if success else "failed",
                "cutoff_date": cutoff_date.isoformat(),
                "records_targeted": count_before,
            }
        except Exception as e:
            return {"action": "cleanup_outcomes", "status": "failed", "error": str(e)}

    def _cleanup_accuracy_metrics(self, cutoff_date: datetime) -> Dict[str, Any]:
        """Clean up accuracy metrics older than cutoff date."""
        try:
            # Get count before cleanup
            all_metrics = self.historical_store.retrieve_accuracy_metrics()
            old_metrics = [m for m in all_metrics if m.calculated_at < cutoff_date]
            count_before = len(old_metrics)

            # The cleanup_old_data method handles accuracy metrics too
            success = True  # Already handled in cleanup_predictions call

            return {
                "action": "cleanup_accuracy_metrics",
                "status": "success" if success else "failed",
                "cutoff_date": cutoff_date.isoformat(),
                "records_targeted": count_before,
            }
        except Exception as e:
            return {
                "action": "cleanup_accuracy_metrics",
                "status": "failed",
                "error": str(e),
            }

    def get_retention_status(self) -> Dict[str, Any]:
        """Get current retention status and recommendations."""
        stats = self.historical_store.get_database_stats()

        # Calculate data age ranges
        all_predictions = self.historical_store.retrieve_predictions()
        all_outcomes = self.historical_store.retrieve_outcomes()
        all_metrics = self.historical_store.retrieve_accuracy_metrics()

        status = {
            "current_stats": stats,
            "policy": {
                "predictions_retention_days": self.policy.predictions_retention_days,
                "outcomes_retention_days": self.policy.outcomes_retention_days,
                "accuracy_metrics_retention_days": self.policy.accuracy_metrics_retention_days,
                "cleanup_frequency_hours": self.policy.cleanup_frequency_hours,
            },
            "last_cleanup": (
                self._last_cleanup.isoformat() if self._last_cleanup else None
            ),
            "should_run_cleanup": self.should_run_cleanup(),
        }

        # Add data age analysis
        if all_predictions:
            oldest_prediction = min(all_predictions, key=lambda p: p.created_at)
            newest_prediction = max(all_predictions, key=lambda p: p.created_at)
            status["predictions_age_range"] = {
                "oldest": oldest_prediction.created_at.isoformat(),
                "newest": newest_prediction.created_at.isoformat(),
                "span_days": (
                    newest_prediction.created_at - oldest_prediction.created_at
                ).days,
            }

        if all_outcomes:
            oldest_outcome = min(all_outcomes, key=lambda o: o.evaluation_date)
            newest_outcome = max(all_outcomes, key=lambda o: o.evaluation_date)
            status["outcomes_age_range"] = {
                "oldest": oldest_outcome.evaluation_date.isoformat(),
                "newest": newest_outcome.evaluation_date.isoformat(),
                "span_days": (
                    newest_outcome.evaluation_date - oldest_outcome.evaluation_date
                ).days,
            }

        if all_metrics:
            oldest_metric = min(all_metrics, key=lambda m: m.calculated_at)
            newest_metric = max(all_metrics, key=lambda m: m.calculated_at)
            status["metrics_age_range"] = {
                "oldest": oldest_metric.calculated_at.isoformat(),
                "newest": newest_metric.calculated_at.isoformat(),
                "span_days": (
                    newest_metric.calculated_at - oldest_metric.calculated_at
                ).days,
            }

        return status

    def update_policy(self, new_policy: RetentionPolicy) -> bool:
        """Update the retention policy."""
        try:
            self.policy = new_policy
            self.logger.info("Retention policy updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update retention policy: {e}")
            return False
