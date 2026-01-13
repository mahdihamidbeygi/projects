"""
Results aggregation component for combining and weighting market predictions.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean, median

from ..interfaces import ResultsAggregator
from ..models import MarketPrediction, NewsArticle, HistoricalAccuracy


logger = logging.getLogger(__name__)


class ResultsAggregatorImpl(ResultsAggregator):
    """Implementation of results aggregation for market predictions."""

    def __init__(self):
        """Initialize the results aggregator."""
        self.logger = logger

    def aggregate_predictions(
        self, predictions: List[MarketPrediction]
    ) -> List[MarketPrediction]:
        """
        Aggregate predictions for multiple stocks and time periods.

        Groups predictions by stock symbol and creates aggregated predictions
        with weighted confidence scores.
        """
        if not predictions:
            return []

        # Group predictions by stock symbol
        stock_groups = defaultdict(list)
        for prediction in predictions:
            stock_groups[prediction.stock_symbol].append(prediction)

        aggregated_predictions = []

        for stock_symbol, stock_predictions in stock_groups.items():
            if len(stock_predictions) == 1:
                # Single prediction, no aggregation needed
                aggregated_predictions.append(stock_predictions[0])
            else:
                # Multiple predictions, aggregate them
                aggregated = self._aggregate_stock_predictions(stock_predictions)
                aggregated_predictions.append(aggregated)

        return aggregated_predictions

    def weight_by_confidence(
        self, predictions: List[MarketPrediction]
    ) -> List[MarketPrediction]:
        """
        Weight predictions by confidence levels.

        Adjusts prediction weights based on confidence levels and returns
        predictions sorted by weighted confidence.
        """
        if not predictions:
            return []

        # Calculate weighted scores for sorting
        weighted_predictions = []
        for prediction in predictions:
            # Weight combines confidence level and impact magnitude
            weighted_score = prediction.confidence_level * prediction.impact_magnitude
            weighted_predictions.append((weighted_score, prediction))

        # Sort by weighted score (descending)
        weighted_predictions.sort(key=lambda x: x[0], reverse=True)

        return [pred for _, pred in weighted_predictions]

    def calculate_accuracy_metrics(
        self, predictions: List[MarketPrediction], actual_outcomes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate accuracy metrics for historical predictions.

        Compares predictions with actual outcomes to calculate various
        accuracy metrics.
        """
        if not predictions or not actual_outcomes:
            return {
                "overall_accuracy": 0.0,
                "directional_accuracy": 0.0,
                "confidence_correlation": 0.0,
                "average_confidence": 0.0,
            }

        # Create lookup for outcomes by prediction
        outcome_lookup = {}
        for outcome in actual_outcomes:
            if "prediction_id" in outcome:
                outcome_lookup[outcome["prediction_id"]] = outcome

        correct_predictions = 0
        directional_correct = 0
        confidence_scores = []
        accuracy_by_confidence = []

        for prediction in predictions:
            prediction_id = getattr(prediction, "id", prediction.article_id)
            outcome = outcome_lookup.get(prediction_id)

            if outcome:
                confidence_scores.append(prediction.confidence_level)

                # Check directional accuracy
                predicted_direction = prediction.impact_direction
                actual_direction = outcome.get("actual_direction", "neutral")

                if predicted_direction == actual_direction:
                    directional_correct += 1

                    # Check magnitude accuracy (within reasonable threshold)
                    predicted_magnitude = prediction.impact_magnitude
                    actual_magnitude = outcome.get("actual_magnitude", 0.0)

                    # Consider prediction correct if direction matches and magnitude is within 30%
                    magnitude_diff = abs(predicted_magnitude - actual_magnitude)
                    if magnitude_diff <= 0.3:
                        correct_predictions += 1
                        accuracy_by_confidence.append(
                            (prediction.confidence_level, 1.0)
                        )
                    else:
                        accuracy_by_confidence.append(
                            (prediction.confidence_level, 0.0)
                        )
                else:
                    accuracy_by_confidence.append((prediction.confidence_level, 0.0))

        total_evaluated = len(
            [p for p in predictions if getattr(p, "id", p.article_id) in outcome_lookup]
        )

        if total_evaluated == 0:
            return {
                "overall_accuracy": 0.0,
                "directional_accuracy": 0.0,
                "confidence_correlation": 0.0,
                "average_confidence": 0.0,
            }

        overall_accuracy = correct_predictions / total_evaluated
        directional_accuracy = directional_correct / total_evaluated
        average_confidence = mean(confidence_scores) if confidence_scores else 0.0

        # Calculate confidence correlation (higher confidence should correlate with higher accuracy)
        confidence_correlation = 0.0
        if len(accuracy_by_confidence) > 1:
            try:
                confidences = [x[0] for x in accuracy_by_confidence]
                accuracies = [x[1] for x in accuracy_by_confidence]

                # Simple correlation calculation
                conf_mean = mean(confidences)
                acc_mean = mean(accuracies)

                numerator = sum(
                    (c - conf_mean) * (a - acc_mean)
                    for c, a in zip(confidences, accuracies)
                )
                conf_var = sum((c - conf_mean) ** 2 for c in confidences)
                acc_var = sum((a - acc_mean) ** 2 for a in accuracies)

                if conf_var > 0 and acc_var > 0:
                    confidence_correlation = numerator / (conf_var * acc_var) ** 0.5
            except (ValueError, ZeroDivisionError):
                confidence_correlation = 0.0

        return {
            "overall_accuracy": overall_accuracy,
            "directional_accuracy": directional_accuracy,
            "confidence_correlation": confidence_correlation,
            "average_confidence": average_confidence,
        }

    def _aggregate_stock_predictions(
        self, predictions: List[MarketPrediction]
    ) -> MarketPrediction:
        """
        Aggregate multiple predictions for the same stock symbol.

        Uses weighted averaging based on confidence levels to create
        a single aggregated prediction.
        """
        if not predictions:
            raise ValueError("Cannot aggregate empty prediction list")

        if len(predictions) == 1:
            return predictions[0]

        # Calculate weighted averages
        total_weight = sum(p.confidence_level for p in predictions)

        if total_weight == 0:
            # All predictions have zero confidence, use simple average
            weights = [1.0 / len(predictions)] * len(predictions)
        else:
            weights = [p.confidence_level / total_weight for p in predictions]

        # Weighted average of impact magnitudes
        weighted_magnitude = sum(
            p.impact_magnitude * w for p, w in zip(predictions, weights)
        )

        # Weighted average of confidence levels
        weighted_confidence = sum(
            p.confidence_level * w for p, w in zip(predictions, weights)
        )

        # Determine aggregated direction based on weighted impact
        direction_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        for prediction, weight in zip(predictions, weights):
            if prediction.impact_direction == "positive":
                direction_scores["positive"] += weight * prediction.impact_magnitude
            elif prediction.impact_direction == "negative":
                direction_scores["negative"] += weight * prediction.impact_magnitude
            else:
                direction_scores["neutral"] += weight

        # Choose direction with highest weighted score
        aggregated_direction = max(direction_scores.items(), key=lambda x: x[1])[0]

        # Combine reasoning from all predictions
        reasoning_parts = []
        for i, prediction in enumerate(predictions):
            reasoning_parts.append(f"P{i+1}: {prediction.reasoning}")

        aggregated_reasoning = (
            f"Aggregated from {len(predictions)} predictions: "
            + "; ".join(reasoning_parts)
        )

        # Use the most recent article_id and creation time
        most_recent = max(predictions, key=lambda p: p.created_at)

        return MarketPrediction(
            article_id=f"AGG_{most_recent.article_id}",
            stock_symbol=predictions[0].stock_symbol,
            impact_direction=aggregated_direction,
            impact_magnitude=weighted_magnitude,
            confidence_level=weighted_confidence,
            reasoning=aggregated_reasoning,
            created_at=datetime.now(),
        )

    def get_stock_summary(
        self,
        predictions: List[MarketPrediction],
        historical_accuracy: Optional[HistoricalAccuracy] = None,
    ) -> Dict[str, Any]:
        """
        Generate a summary for a specific stock's predictions.

        Returns aggregated metrics and historical context for a stock.
        """
        if not predictions:
            return {}

        stock_symbol = predictions[0].stock_symbol

        # Basic statistics
        confidence_levels = [p.confidence_level for p in predictions]
        impact_magnitudes = [p.impact_magnitude for p in predictions]

        # Direction distribution
        directions = [p.impact_direction for p in predictions]
        direction_counts = {
            "positive": directions.count("positive"),
            "negative": directions.count("negative"),
            "neutral": directions.count("neutral"),
        }

        summary = {
            "stock_symbol": stock_symbol,
            "total_predictions": len(predictions),
            "average_confidence": mean(confidence_levels),
            "median_confidence": median(confidence_levels),
            "average_impact_magnitude": mean(impact_magnitudes),
            "direction_distribution": direction_counts,
            "highest_confidence_prediction": max(
                predictions, key=lambda p: p.confidence_level
            ),
            "most_recent_prediction": max(predictions, key=lambda p: p.created_at),
        }

        # Add historical accuracy if available
        if historical_accuracy:
            summary["historical_accuracy"] = {
                "accuracy_rate": historical_accuracy.accuracy_rate,
                "total_historical_predictions": historical_accuracy.total_predictions,
                "time_period_days": historical_accuracy.time_period_days,
            }

        return summary
