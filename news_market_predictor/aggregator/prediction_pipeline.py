"""
Prediction pipeline that integrates aggregation and display functionality.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..models import (
    MarketPrediction,
    NewsArticle,
    ExtractedEntity,
    SentimentAnalysis,
    HistoricalAccuracy,
)
from .results_aggregator import ResultsAggregatorImpl
from .display_formatter import DisplayFormatter


logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Complete pipeline for processing, aggregating, and displaying market predictions.

    Integrates all components needed for the results aggregation and display system.
    """

    def __init__(self):
        """Initialize the prediction pipeline."""
        self.aggregator = ResultsAggregatorImpl()
        self.formatter = DisplayFormatter()
        self.logger = logger

    def process_predictions(
        self,
        predictions: List[MarketPrediction],
        articles: Optional[Dict[str, NewsArticle]] = None,
        entities: Optional[Dict[str, List[ExtractedEntity]]] = None,
        sentiments: Optional[Dict[str, SentimentAnalysis]] = None,
        historical_accuracies: Optional[Dict[str, HistoricalAccuracy]] = None,
        aggregate_by_stock: bool = True,
    ) -> Tuple[List[MarketPrediction], List[Dict[str, Any]]]:
        """
        Process predictions through the complete pipeline.

        Args:
            predictions: List of market predictions to process
            articles: Dictionary mapping article_id to NewsArticle objects
            entities: Dictionary mapping article_id to lists of ExtractedEntity objects
            sentiments: Dictionary mapping article_id to SentimentAnalysis objects
            historical_accuracies: Dictionary mapping stock_symbol to HistoricalAccuracy objects
            aggregate_by_stock: Whether to aggregate multiple predictions per stock

        Returns:
            Tuple of (processed_predictions, formatted_display_data)
        """
        try:
            self.logger.info(
                f"Processing {len(predictions)} predictions through pipeline"
            )

            # Step 1: Aggregate predictions if requested
            if aggregate_by_stock:
                processed_predictions = self.aggregator.aggregate_predictions(
                    predictions
                )
                self.logger.info(
                    f"Aggregated to {len(processed_predictions)} predictions"
                )
            else:
                processed_predictions = predictions

            # Step 2: Weight by confidence
            processed_predictions = self.aggregator.weight_by_confidence(
                processed_predictions
            )

            # Step 3: Format for display
            formatted_data = self.formatter.format_aggregated_predictions(
                predictions=processed_predictions,
                articles=articles,
                entities=entities,
                sentiments=sentiments,
                historical_accuracies=historical_accuracies,
            )

            self.logger.info("Successfully processed predictions through pipeline")
            return processed_predictions, formatted_data

        except Exception as e:
            self.logger.error(f"Error processing predictions: {e}")
            raise

    def export_results(
        self,
        predictions: List[MarketPrediction],
        articles: Optional[Dict[str, NewsArticle]] = None,
        entities: Optional[Dict[str, List[ExtractedEntity]]] = None,
        sentiments: Optional[Dict[str, SentimentAnalysis]] = None,
        historical_accuracies: Optional[Dict[str, HistoricalAccuracy]] = None,
        export_format: str = "json",
        filename: Optional[str] = None,
    ) -> str:
        """
        Export predictions in the specified format.

        Args:
            predictions: List of predictions to export
            articles: Article data for context
            entities: Entity data for context
            sentiments: Sentiment data for context
            historical_accuracies: Historical accuracy data
            export_format: "json" or "csv"
            filename: Optional filename to save to

        Returns:
            Exported data as string
        """
        try:
            if export_format.lower() == "json":
                return self.formatter.export_to_json(
                    predictions=predictions,
                    articles=articles,
                    entities=entities,
                    sentiments=sentiments,
                    historical_accuracies=historical_accuracies,
                    filename=filename,
                )
            elif export_format.lower() == "csv":
                return self.formatter.export_to_csv(
                    predictions=predictions,
                    articles=articles,
                    entities=entities,
                    sentiments=sentiments,
                    historical_accuracies=historical_accuracies,
                    filename=filename,
                )
            else:
                raise ValueError(f"Unsupported export format: {export_format}")

        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            raise

    def generate_summary_report(
        self,
        predictions: List[MarketPrediction],
        actual_outcomes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.

        Args:
            predictions: List of predictions to summarize
            actual_outcomes: Optional actual market outcomes for accuracy calculation

        Returns:
            Summary report dictionary
        """
        try:
            # Calculate accuracy metrics if outcomes provided
            accuracy_metrics = None
            if actual_outcomes:
                accuracy_metrics = self.aggregator.calculate_accuracy_metrics(
                    predictions, actual_outcomes
                )

            # Generate summary report
            summary = self.formatter.create_summary_report(
                predictions=predictions, accuracy_metrics=accuracy_metrics
            )

            return summary

        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            raise

    def get_stock_insights(
        self,
        predictions: List[MarketPrediction],
        stock_symbol: str,
        historical_accuracy: Optional[HistoricalAccuracy] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed insights for a specific stock.

        Args:
            predictions: All predictions to analyze
            stock_symbol: Stock symbol to focus on
            historical_accuracy: Historical accuracy data for the stock

        Returns:
            Detailed insights dictionary for the stock
        """
        try:
            # Filter predictions for the specific stock
            stock_predictions = [
                p for p in predictions if p.stock_symbol == stock_symbol
            ]

            if not stock_predictions:
                return {"error": f"No predictions found for stock {stock_symbol}"}

            # Get stock summary from aggregator
            summary = self.aggregator.get_stock_summary(
                predictions=stock_predictions, historical_accuracy=historical_accuracy
            )

            # Add additional insights
            summary["insights"] = {
                "prediction_trend": self._analyze_prediction_trend(stock_predictions),
                "confidence_analysis": self._analyze_confidence_pattern(
                    stock_predictions
                ),
                "risk_assessment": self._assess_risk_level(
                    stock_predictions, historical_accuracy
                ),
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating stock insights: {e}")
            raise

    def _analyze_prediction_trend(self, predictions: List[MarketPrediction]) -> str:
        """Analyze the trend in predictions over time."""
        if len(predictions) < 2:
            return "insufficient_data"

        # Sort by creation time
        sorted_predictions = sorted(predictions, key=lambda p: p.created_at)

        # Analyze direction trend
        recent_half = sorted_predictions[len(sorted_predictions) // 2 :]
        positive_count = sum(1 for p in recent_half if p.impact_direction == "positive")
        negative_count = sum(1 for p in recent_half if p.impact_direction == "negative")

        if positive_count > negative_count * 1.5:
            return "increasingly_bullish"
        elif negative_count > positive_count * 1.5:
            return "increasingly_bearish"
        else:
            return "mixed_signals"

    def _analyze_confidence_pattern(
        self, predictions: List[MarketPrediction]
    ) -> Dict[str, Any]:
        """Analyze confidence patterns in predictions."""
        confidences = [p.confidence_level for p in predictions]

        return {
            "average_confidence": sum(confidences) / len(confidences),
            "confidence_stability": (
                "stable" if max(confidences) - min(confidences) < 0.3 else "variable"
            ),
            "high_confidence_predictions": len([c for c in confidences if c >= 0.7]),
            "low_confidence_predictions": len([c for c in confidences if c < 0.3]),
        }

    def _assess_risk_level(
        self,
        predictions: List[MarketPrediction],
        historical_accuracy: Optional[HistoricalAccuracy],
    ) -> str:
        """Assess overall risk level for the stock."""
        # Base risk on prediction consistency and historical accuracy
        avg_confidence = sum(p.confidence_level for p in predictions) / len(predictions)

        # Check prediction consistency
        directions = [p.impact_direction for p in predictions]
        most_common_direction = max(set(directions), key=directions.count)
        consistency_ratio = directions.count(most_common_direction) / len(directions)

        # Factor in historical accuracy if available
        historical_factor = 1.0
        if historical_accuracy and historical_accuracy.total_predictions > 10:
            historical_factor = historical_accuracy.accuracy_rate

        # Calculate risk score
        risk_score = avg_confidence * consistency_ratio * historical_factor

        if risk_score >= 0.7:
            return "low_risk"
        elif risk_score >= 0.4:
            return "medium_risk"
        else:
            return "high_risk"
