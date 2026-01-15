"""
Standalone market prediction engine for testing.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import directly without relative imports
from news_market_predictor.interfaces import MarketPredictor
from news_market_predictor.models import (
    NewsArticle,
    SentimentAnalysis,
    ExtractedEntity,
    MarketPrediction,
)
from news_market_predictor.error_handling import (
    InvalidInputHandler,
    with_error_recovery,
    ErrorHandlingManager,
)
from news_market_predictor.exceptions import PredictionError


logger = logging.getLogger(__name__)


@dataclass
class PredictionSignal:
    """Internal representation of a prediction signal."""

    signal_type: str  # sentiment, entity, financial_metric
    signal_strength: float  # 0.0 to 1.0
    direction: str  # positive, negative, neutral
    confidence: float  # 0.0 to 1.0
    source_data: Dict[str, Any]


class BasicMarketPredictor(MarketPredictor):
    """
    Basic implementation of market prediction engine.

    Uses sentiment analysis and entity extraction to generate predictions
    about stock market impact from news articles.
    """

    def __init__(self, historical_accuracy: float = 0.65, historical_analyzer=None):
        """
        Initialize the market predictor.

        Args:
            historical_accuracy: Base historical accuracy for confidence calculation
            historical_analyzer: Optional HistoricalAnalyzer for incorporating historical data
        """
        self.historical_accuracy = historical_accuracy
        self.historical_analyzer = historical_analyzer
        self.low_confidence_threshold = 0.30

        # Weights for different signal types
        self.signal_weights = {"sentiment": 0.4, "entity": 0.3, "financial_metric": 0.3}

        # Setup error handling
        self.error_manager = ErrorHandlingManager()
        self.input_handler = InvalidInputHandler()

        # Keywords that indicate strong market impact
        self.impact_keywords = {
            "positive": [
                "earnings beat",
                "revenue growth",
                "profit increase",
                "upgrade",
                "acquisition",
                "partnership",
                "expansion",
                "breakthrough",
                "record high",
                "outperform",
                "bullish",
                "buy rating",
            ],
            "negative": [
                "earnings miss",
                "revenue decline",
                "loss",
                "downgrade",
                "lawsuit",
                "investigation",
                "bankruptcy",
                "layoffs",
                "recall",
                "scandal",
                "bearish",
                "sell rating",
            ],
        }

    def predict_impact(
        self,
        article: NewsArticle,
        sentiment: SentimentAnalysis,
        entities: List[ExtractedEntity],
    ) -> List[MarketPrediction]:
        """
        Generate market impact predictions for all stocks mentioned in the article.
        """
        try:
            # Validate inputs first
            if not self.input_handler.validate_article(article):
                logger.warning(
                    "Invalid article input for prediction: %s",
                    article.id if article else "None",
                )
                return []

            if not self.input_handler.validate_sentiment(sentiment):
                logger.warning("Invalid sentiment input for article: %s", article.id)
                # Create neutral prediction for invalid sentiment
                stock_entities = [
                    e for e in entities if e.entity_type == "stock_symbol"
                ]
                return [
                    self.input_handler.create_neutral_prediction(
                        article.id,
                        entity.entity_value,
                        "Invalid sentiment analysis data",
                    )
                    for entity in stock_entities
                ]

            if not self.input_handler.validate_entities(entities):
                logger.warning("Invalid entities input for article: %s", article.id)
                return []

            # Extract stock symbols from entities
            stock_entities = [e for e in entities if e.entity_type == "stock_symbol"]

            if not stock_entities:
                logger.info("No stock symbols found in article %s", article.id)
                return []

            predictions = []

            # Generate prediction for each stock symbol
            for stock_entity in stock_entities:
                try:
                    base_prediction = self._generate_single_prediction(
                        article, sentiment, entities, stock_entity
                    )

                    # Apply historical influence if analyzer is available
                    if self.historical_analyzer:
                        try:
                            influenced_prediction = (
                                self.historical_analyzer.calculate_historical_influence(
                                    stock_symbol=stock_entity.entity_value,
                                    sentiment=sentiment,
                                    base_prediction=base_prediction,
                                )
                            )
                            predictions.append(influenced_prediction)
                        except Exception as e:
                            logger.warning(
                                "Failed to apply historical influence for %s: %s. Using base prediction.",
                                stock_entity.entity_value,
                                e,
                            )
                            predictions.append(base_prediction)
                    else:
                        predictions.append(base_prediction)

                except Exception as e:
                    logger.error(
                        "Error generating prediction for %s: %s",
                        stock_entity.entity_value,
                        e,
                    )
                    # Create neutral prediction as fallback
                    neutral_prediction = self.input_handler.create_neutral_prediction(
                        article.id,
                        stock_entity.entity_value,
                        f"Error in prediction generation: {str(e)}",
                    )
                    predictions.append(neutral_prediction)

            return predictions

        except Exception as e:
            logger.error(
                "Error in predict_impact for article %s: %s",
                article.id if article else "None",
                e,
            )
            return []

    def _generate_single_prediction(
        self,
        article: NewsArticle,
        sentiment: SentimentAnalysis,
        entities: List[ExtractedEntity],
        stock_entity: ExtractedEntity,
    ) -> MarketPrediction:
        """Generate a single prediction for a specific stock."""

        # Collect prediction signals
        signals = self._collect_prediction_signals(
            article, sentiment, entities, stock_entity
        )

        # Aggregate signals to determine impact
        aggregated_signal = self.aggregate_signals(signals)

        # Calculate confidence based on signal strength and historical accuracy
        confidence = self.calculate_confidence(aggregated_signal)

        # Determine impact direction and magnitude
        impact_direction = aggregated_signal.get("direction", "neutral")
        impact_magnitude = aggregated_signal.get("magnitude", 0.0)

        # Generate reasoning
        reasoning = self._generate_reasoning(signals, sentiment, stock_entity)

        # Flag low confidence predictions
        if confidence < self.low_confidence_threshold:
            reasoning += f" [LOW CONFIDENCE: {confidence:.2f}]"

        return MarketPrediction(
            article_id=article.id,
            stock_symbol=stock_entity.entity_value,
            impact_direction=impact_direction,
            impact_magnitude=impact_magnitude,
            confidence_level=confidence,
            reasoning=reasoning,
            created_at=datetime.now(),
        )

    def _collect_prediction_signals(
        self,
        article: NewsArticle,
        sentiment: SentimentAnalysis,
        entities: List[ExtractedEntity],
        stock_entity: ExtractedEntity,
    ) -> List[Dict[str, Any]]:
        """Collect all prediction signals for analysis."""

        signals = []

        # Sentiment signal
        sentiment_signal = {
            "type": "sentiment",
            "strength": abs(sentiment.sentiment_score),
            "direction": self._sentiment_to_direction(sentiment.sentiment_score),
            "confidence": sentiment.confidence,
            "data": {
                "sentiment_score": sentiment.sentiment_score,
                "market_tone": sentiment.market_tone,
                "key_phrases": sentiment.key_phrases,
            },
        }
        signals.append(sentiment_signal)

        # Entity relevance signal
        entity_signal = {
            "type": "entity",
            "strength": stock_entity.relevance_score,
            "direction": "neutral",  # Entity presence doesn't indicate direction
            "confidence": stock_entity.relevance_score,
            "data": {
                "entity_value": stock_entity.entity_value,
                "context": stock_entity.context,
            },
        }
        signals.append(entity_signal)

        # Financial metrics signal
        financial_entities = [e for e in entities if e.entity_type == "metric"]
        if financial_entities:
            avg_relevance = sum(e.relevance_score for e in financial_entities) / len(
                financial_entities
            )
            financial_signal = {
                "type": "financial_metric",
                "strength": avg_relevance,
                "direction": self._analyze_financial_metrics_direction(article.content),
                "confidence": avg_relevance,
                "data": {
                    "metrics": [e.entity_value for e in financial_entities],
                    "count": len(financial_entities),
                },
            }
            signals.append(financial_signal)

        # Keyword impact signal
        keyword_signal = self._analyze_impact_keywords(article.content)
        if keyword_signal:
            signals.append(keyword_signal)

        return signals

    def _sentiment_to_direction(self, sentiment_score: float) -> str:
        """Convert sentiment score to impact direction."""
        if sentiment_score > 0.1:
            return "positive"
        if sentiment_score < -0.1:
            return "negative"
        return "neutral"

    def _analyze_financial_metrics_direction(self, content: str) -> str:
        """Analyze financial metrics to determine impact direction."""

        positive_indicators = ["increase", "growth", "beat", "exceed", "higher", "up"]
        negative_indicators = ["decrease", "decline", "miss", "lower", "down", "fall"]

        content_lower = content.lower()

        positive_count = sum(
            1 for indicator in positive_indicators if indicator in content_lower
        )
        negative_count = sum(
            1 for indicator in negative_indicators if indicator in content_lower
        )

        if positive_count > negative_count:
            return "positive"
        if negative_count > positive_count:
            return "negative"
        return "neutral"

    def _analyze_impact_keywords(self, content: str) -> Optional[Dict[str, Any]]:
        """Analyze content for high-impact keywords."""

        content_lower = content.lower()

        positive_matches = sum(
            1
            for keyword in self.impact_keywords["positive"]
            if keyword in content_lower
        )
        negative_matches = sum(
            1
            for keyword in self.impact_keywords["negative"]
            if keyword in content_lower
        )

        total_matches = positive_matches + negative_matches

        if total_matches == 0:
            return None

        # Determine direction and strength
        if positive_matches > negative_matches:
            direction = "positive"
            strength = positive_matches / len(self.impact_keywords["positive"])
        elif negative_matches > positive_matches:
            direction = "negative"
            strength = negative_matches / len(self.impact_keywords["negative"])
        else:
            direction = "neutral"
            strength = 0.5

        return {
            "type": "keyword",
            "strength": min(strength, 1.0),
            "direction": direction,
            "confidence": min(total_matches / 5.0, 1.0),  # Max confidence at 5+ matches
            "data": {
                "positive_matches": positive_matches,
                "negative_matches": negative_matches,
                "total_matches": total_matches,
            },
        }

    def calculate_confidence(self, prediction_data: Dict[str, Any]) -> float:
        """
        Calculate confidence level for predictions based on signal strength and historical accuracy.
        """
        try:
            # Base confidence from historical accuracy
            base_confidence = self.historical_accuracy

            # Adjust based on signal strength
            signal_strength = prediction_data.get("strength", 0.0)
            signal_confidence = prediction_data.get("confidence", 0.0)

            # Combine factors using weighted average
            confidence_factors = [
                (base_confidence, 0.3),  # Historical accuracy weight
                (signal_strength, 0.4),  # Signal strength weight
                (signal_confidence, 0.3),  # Signal confidence weight
            ]

            weighted_confidence = sum(
                factor * weight for factor, weight in confidence_factors
            )

            # Apply penalty for neutral predictions (less confident)
            if prediction_data.get("direction") == "neutral":
                weighted_confidence *= 0.8

            # Ensure confidence is within bounds
            return max(0.0, min(1.0, weighted_confidence))

        except Exception as e:
            logger.error("Error calculating confidence: %s", e)
            return 0.0

    def aggregate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple prediction signals into a single prediction.
        """
        try:
            if not signals:
                return {
                    "direction": "neutral",
                    "magnitude": 0.0,
                    "strength": 0.0,
                    "confidence": 0.0,
                }

            # Calculate weighted scores for each direction
            positive_score = 0.0
            negative_score = 0.0
            total_weight = 0.0
            total_confidence = 0.0

            for signal in signals:
                signal_type = signal.get("type", "unknown")
                weight = self.signal_weights.get(signal_type, 0.1)
                strength = signal.get("strength", 0.0)
                direction = signal.get("direction", "neutral")
                confidence = signal.get("confidence", 0.0)

                weighted_strength = strength * weight * confidence

                if direction == "positive":
                    positive_score += weighted_strength
                elif direction == "negative":
                    negative_score += weighted_strength

                total_weight += weight
                total_confidence += confidence

            # Normalize scores
            if total_weight > 0:
                positive_score /= total_weight
                negative_score /= total_weight

            # Determine overall direction and magnitude
            net_score = positive_score - negative_score
            magnitude = abs(net_score)

            if net_score > 0.1:
                direction = "positive"
            elif net_score < -0.1:
                direction = "negative"
            else:
                direction = "neutral"
                magnitude = 0.0

            # Calculate overall confidence
            avg_confidence = total_confidence / len(signals) if signals else 0.0

            return {
                "direction": direction,
                "magnitude": min(magnitude, 1.0),
                "strength": magnitude,
                "confidence": avg_confidence,
                "positive_score": positive_score,
                "negative_score": negative_score,
                "signal_count": len(signals),
            }

        except Exception as e:
            logger.error("Error aggregating signals: %s", e)
            return {
                "direction": "neutral",
                "magnitude": 0.0,
                "strength": 0.0,
                "confidence": 0.0,
            }

    def _generate_reasoning(
        self,
        signals: List[Dict[str, Any]],
        sentiment: SentimentAnalysis,
        stock_entity: ExtractedEntity,
    ) -> str:
        """Generate human-readable reasoning for the prediction."""

        try:
            reasoning_parts = []

            # Add sentiment reasoning
            if sentiment.sentiment_score != 0:
                tone_desc = f"{sentiment.market_tone} sentiment (score: {sentiment.sentiment_score:.2f})"
                reasoning_parts.append(f"Article shows {tone_desc}")

            # Add entity relevance
            if stock_entity.relevance_score > 0.5:
                reasoning_parts.append(
                    f"High relevance to {stock_entity.entity_value} (score: {stock_entity.relevance_score:.2f})"
                )

            # Add signal summary
            signal_types = [s.get("type") for s in signals]
            if len(signal_types) > 1:
                reasoning_parts.append(
                    f"Based on {len(signal_types)} analysis factors: {', '.join(set(signal_types))}"
                )

            # Add key phrases if available
            if sentiment.key_phrases:
                key_phrases_str = ", ".join(
                    sentiment.key_phrases[:3]
                )  # Limit to first 3
                reasoning_parts.append(f"Key phrases: {key_phrases_str}")

            if reasoning_parts:
                return ". ".join(reasoning_parts) + "."

            return f"Analysis of {stock_entity.entity_value} based on article content."

        except Exception as e:
            logger.error("Error generating reasoning: %s", e)
            return f"Prediction for {stock_entity.entity_value} based on automated analysis."


if __name__ == "__main__":
    # Test the implementation
    print("Testing BasicMarketPredictor...")
    predictor = BasicMarketPredictor()
    print("✓ BasicMarketPredictor created successfully")
    print("Market prediction engine implementation completed!")
