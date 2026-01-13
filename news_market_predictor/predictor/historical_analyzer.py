"""
Historical analysis component for incorporating historical data into current predictions.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import statistics

from ..models import (
    MarketPrediction,
    MarketOutcome,
    HistoricalAccuracy,
    SentimentAnalysis,
)
from ..storage.historical_data_store import HistoricalDataStore


class HistoricalAnalyzer:
    """Analyzes historical data to influence current predictions."""

    def __init__(self, historical_store: HistoricalDataStore):
        """Initialize with historical data store."""
        self.historical_store = historical_store

    def calculate_historical_influence(
        self,
        stock_symbol: str,
        sentiment: SentimentAnalysis,
        base_prediction: MarketPrediction,
    ) -> MarketPrediction:
        """
        Incorporate historical data influence into current prediction.

        Args:
            stock_symbol: The stock symbol being predicted
            sentiment: Current sentiment analysis
            base_prediction: Base prediction before historical influence

        Returns:
            Modified prediction incorporating historical data
        """
        # Get historical accuracy for this stock
        historical_accuracy = self.historical_store.calculate_historical_accuracy(
            stock_symbol=stock_symbol, time_period_days=30
        )

        # Get similar historical predictions
        similar_predictions = self.historical_store.get_similar_historical_predictions(
            stock_symbol=stock_symbol,
            sentiment_score=sentiment.sentiment_score,
            lookback_days=90,
        )

        # Calculate influence factors
        accuracy_factor = self._calculate_accuracy_factor(historical_accuracy)
        similarity_factor = self._calculate_similarity_factor(
            similar_predictions, sentiment
        )

        # Adjust confidence based on historical performance
        adjusted_confidence = self._adjust_confidence(
            base_prediction.confidence_level, accuracy_factor, similarity_factor
        )

        # Adjust impact direction and magnitude based on historical patterns
        adjusted_direction, adjusted_magnitude = self._adjust_prediction_values(
            base_prediction, similar_predictions, similarity_factor
        )

        # Create new prediction with historical influence
        influenced_prediction = MarketPrediction(
            article_id=base_prediction.article_id,
            stock_symbol=base_prediction.stock_symbol,
            impact_direction=adjusted_direction,
            impact_magnitude=adjusted_magnitude,
            confidence_level=adjusted_confidence,
            reasoning=self._create_historical_reasoning(
                base_prediction.reasoning,
                historical_accuracy,
                similar_predictions,
                accuracy_factor,
                similarity_factor,
            ),
            created_at=base_prediction.created_at,
        )

        return influenced_prediction

    def _calculate_accuracy_factor(
        self, accuracy: Optional[HistoricalAccuracy]
    ) -> float:
        """
        Calculate accuracy factor based on historical performance.

        Returns:
            Factor between 0.5 and 1.5 to adjust confidence
        """
        if not accuracy or accuracy.total_predictions < 5:
            # Not enough historical data, use neutral factor
            return 1.0

        # Scale accuracy rate to influence factor
        # 0.0 accuracy -> 0.5 factor (reduce confidence)
        # 0.5 accuracy -> 1.0 factor (neutral)
        # 1.0 accuracy -> 1.5 factor (increase confidence)
        return 0.5 + accuracy.accuracy_rate

    def _calculate_similarity_factor(
        self,
        similar_predictions: List[Tuple[MarketPrediction, Optional[MarketOutcome]]],
        current_sentiment: SentimentAnalysis,
    ) -> float:
        """
        Calculate similarity factor based on historical predictions with similar sentiment.

        Returns:
            Factor between 0.7 and 1.3 to adjust prediction
        """
        if not similar_predictions:
            return 1.0

        # Calculate accuracy of similar predictions
        correct_similar = 0
        total_similar = 0

        for prediction, outcome in similar_predictions:
            if outcome:
                total_similar += 1
                if prediction.impact_direction == outcome.actual_direction:
                    correct_similar += 1

        if total_similar == 0:
            return 1.0

        similarity_accuracy = correct_similar / total_similar

        # Scale to factor range
        return 0.7 + (similarity_accuracy * 0.6)

    def _adjust_confidence(
        self, base_confidence: float, accuracy_factor: float, similarity_factor: float
    ) -> float:
        """
        Adjust confidence level based on historical factors.

        Args:
            base_confidence: Original confidence level
            accuracy_factor: Factor based on overall historical accuracy
            similarity_factor: Factor based on similar predictions

        Returns:
            Adjusted confidence level (clamped to 0.0-1.0)
        """
        # Combine factors with weights
        combined_factor = (accuracy_factor * 0.6) + (similarity_factor * 0.4)

        # Apply factor to base confidence
        adjusted_confidence = base_confidence * combined_factor

        # Clamp to valid range
        return max(0.0, min(1.0, adjusted_confidence))

    def _adjust_prediction_values(
        self,
        base_prediction: MarketPrediction,
        similar_predictions: List[Tuple[MarketPrediction, Optional[MarketOutcome]]],
        similarity_factor: float,
    ) -> Tuple[str, float]:
        """
        Adjust prediction direction and magnitude based on historical patterns.

        Returns:
            Tuple of (adjusted_direction, adjusted_magnitude)
        """
        if not similar_predictions:
            return base_prediction.impact_direction, base_prediction.impact_magnitude

        # Analyze historical patterns
        direction_counts = {"positive": 0, "negative": 0, "neutral": 0}
        magnitudes = []

        for prediction, outcome in similar_predictions:
            if outcome:
                direction_counts[outcome.actual_direction] += 1
                magnitudes.append(outcome.actual_magnitude)

        # Determine if there's a strong historical pattern
        total_outcomes = sum(direction_counts.values())
        if total_outcomes == 0:
            return base_prediction.impact_direction, base_prediction.impact_magnitude

        # Find dominant direction
        dominant_direction = max(direction_counts, key=direction_counts.get)
        dominant_ratio = direction_counts[dominant_direction] / total_outcomes

        # Adjust direction if historical pattern is strong (>60%)
        adjusted_direction = base_prediction.impact_direction
        if dominant_ratio > 0.6 and similarity_factor > 1.0:
            adjusted_direction = dominant_direction

        # Adjust magnitude based on historical average
        adjusted_magnitude = base_prediction.impact_magnitude
        if magnitudes:
            historical_avg_magnitude = statistics.mean(magnitudes)
            # Blend base prediction with historical average
            blend_weight = min(0.3, (similarity_factor - 1.0) * 0.5)
            adjusted_magnitude = (
                base_prediction.impact_magnitude * (1 - blend_weight)
                + historical_avg_magnitude * blend_weight
            )
            # Clamp to valid range
            adjusted_magnitude = max(0.0, min(1.0, adjusted_magnitude))

        return adjusted_direction, adjusted_magnitude

    def _create_historical_reasoning(
        self,
        base_reasoning: str,
        accuracy: Optional[HistoricalAccuracy],
        similar_predictions: List[Tuple[MarketPrediction, Optional[MarketOutcome]]],
        accuracy_factor: float,
        similarity_factor: float,
    ) -> str:
        """Create enhanced reasoning that includes historical context."""
        historical_context = []

        if accuracy and accuracy.total_predictions >= 5:
            historical_context.append(
                f"Historical accuracy for this stock: {accuracy.accuracy_rate:.1%} "
                f"over {accuracy.total_predictions} predictions in the last 30 days"
            )

        if similar_predictions:
            outcomes_with_data = [p for p, o in similar_predictions if o is not None]
            if outcomes_with_data:
                historical_context.append(
                    f"Based on {len(outcomes_with_data)} similar historical predictions"
                )

        if accuracy_factor != 1.0:
            if accuracy_factor > 1.0:
                historical_context.append(
                    "Confidence increased due to strong historical performance"
                )
            else:
                historical_context.append(
                    "Confidence reduced due to poor historical performance"
                )

        if similarity_factor != 1.0:
            if similarity_factor > 1.0:
                historical_context.append(
                    "Prediction supported by similar historical cases"
                )
            else:
                historical_context.append(
                    "Prediction conflicts with similar historical cases"
                )

        # Combine base reasoning with historical context
        if historical_context:
            return f"{base_reasoning}. Historical analysis: {'; '.join(historical_context)}."
        else:
            return f"{base_reasoning}. No significant historical data available for adjustment."

    def get_prediction_accuracy_trend(
        self, stock_symbol: str, days_back: int = 90
    ) -> Dict[str, Any]:
        """
        Get accuracy trend for a stock over time.

        Returns:
            Dictionary with trend analysis data
        """
        # Calculate accuracy for different time periods
        periods = [7, 14, 30, 60, 90]
        trend_data = {}

        for period in periods:
            if period <= days_back:
                accuracy = self.historical_store.calculate_historical_accuracy(
                    stock_symbol=stock_symbol, time_period_days=period
                )
                if accuracy:
                    trend_data[f"{period}_days"] = {
                        "accuracy_rate": accuracy.accuracy_rate,
                        "total_predictions": accuracy.total_predictions,
                        "average_confidence": accuracy.average_confidence,
                    }

        # Calculate trend direction
        if len(trend_data) >= 2:
            recent_accuracy = trend_data.get("7_days", {}).get("accuracy_rate", 0)
            older_accuracy = trend_data.get("30_days", {}).get("accuracy_rate", 0)

            if recent_accuracy > older_accuracy + 0.1:
                trend_direction = "improving"
            elif recent_accuracy < older_accuracy - 0.1:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"

        return {
            "stock_symbol": stock_symbol,
            "trend_direction": trend_direction,
            "period_data": trend_data,
            "analysis_date": datetime.now().isoformat(),
        }

    def recommend_confidence_adjustment(
        self, stock_symbol: str, base_confidence: float
    ) -> Dict[str, Any]:
        """
        Recommend confidence adjustment based on historical performance.

        Returns:
            Dictionary with recommendation details
        """
        accuracy = self.historical_store.calculate_historical_accuracy(
            stock_symbol=stock_symbol, time_period_days=30
        )

        if not accuracy or accuracy.total_predictions < 5:
            return {
                "recommended_confidence": base_confidence,
                "adjustment_factor": 1.0,
                "reason": "Insufficient historical data for adjustment",
                "data_points": 0,
            }

        # Calculate recommended adjustment
        accuracy_factor = self._calculate_accuracy_factor(accuracy)
        recommended_confidence = min(1.0, max(0.0, base_confidence * accuracy_factor))

        # Determine reason
        if accuracy_factor > 1.1:
            reason = (
                f"Strong historical performance ({accuracy.accuracy_rate:.1%} accuracy)"
            )
        elif accuracy_factor < 0.9:
            reason = (
                f"Poor historical performance ({accuracy.accuracy_rate:.1%} accuracy)"
            )
        else:
            reason = f"Average historical performance ({accuracy.accuracy_rate:.1%} accuracy)"

        return {
            "recommended_confidence": recommended_confidence,
            "adjustment_factor": accuracy_factor,
            "reason": reason,
            "data_points": accuracy.total_predictions,
            "historical_accuracy": accuracy.accuracy_rate,
        }
