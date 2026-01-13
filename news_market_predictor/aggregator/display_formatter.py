"""
Display formatting component for presenting market predictions and results.
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from io import StringIO

from ..models import (
    MarketPrediction,
    NewsArticle,
    HistoricalAccuracy,
    ExtractedEntity,
    SentimentAnalysis,
)


class DisplayFormatter:
    """Formats market predictions and results for display and export."""

    def __init__(self):
        """Initialize the display formatter."""
        pass

    def format_prediction_display(
        self,
        prediction: MarketPrediction,
        article: Optional[NewsArticle] = None,
        entities: Optional[List[ExtractedEntity]] = None,
        sentiment: Optional[SentimentAnalysis] = None,
        historical_accuracy: Optional[HistoricalAccuracy] = None,
    ) -> Dict[str, Any]:
        """
        Format a single prediction for display with all required fields.

        Requirements 4.1, 4.2: Display predictions with impact, confidence, stock symbols,
        article title, and key extracted information.
        """
        display_data = {
            # Core prediction data (Requirement 4.1)
            "impact_prediction": {
                "direction": prediction.impact_direction,
                "magnitude": prediction.impact_magnitude,
                "confidence_level": prediction.confidence_level,
            },
            "stock_symbol": prediction.stock_symbol,
            "prediction_id": prediction.article_id,
            "created_at": prediction.created_at.isoformat(),
            "reasoning": prediction.reasoning,
        }

        # Add article information (Requirement 4.2)
        if article:
            display_data["article_info"] = {
                "title": article.title,
                "published_at": article.published_at.isoformat(),
                "source": article.source,
                "category": article.category,
                "url": article.url,
            }

        # Add key extracted information (Requirement 4.2)
        # Always include extracted_entities section for completeness, even if empty
        if entities is not None:
            display_data["extracted_entities"] = {
                "stock_symbols": [
                    {"symbol": e.entity_value, "relevance": e.relevance_score}
                    for e in entities
                    if e.entity_type == "stock_symbol"
                ],
                "companies": [
                    {"name": e.entity_value, "relevance": e.relevance_score}
                    for e in entities
                    if e.entity_type == "company"
                ],
                "financial_metrics": [
                    {
                        "metric": e.entity_value,
                        "context": e.context,
                        "relevance": e.relevance_score,
                    }
                    for e in entities
                    if e.entity_type == "metric"
                ],
            }

        # Add sentiment analysis
        if sentiment:
            display_data["sentiment_analysis"] = {
                "sentiment_score": sentiment.sentiment_score,
                "market_tone": sentiment.market_tone,
                "confidence": sentiment.confidence,
                "key_phrases": sentiment.key_phrases,
            }

        # Add historical accuracy (Requirement 4.3)
        if historical_accuracy:
            display_data["historical_accuracy"] = {
                "accuracy_rate": historical_accuracy.accuracy_rate,
                "total_predictions": historical_accuracy.total_predictions,
                "time_period_days": historical_accuracy.time_period_days,
                "calculated_at": historical_accuracy.calculated_at.isoformat(),
            }

        return display_data

    def format_aggregated_predictions(
        self,
        predictions: List[MarketPrediction],
        articles: Optional[Dict[str, NewsArticle]] = None,
        entities: Optional[Dict[str, List[ExtractedEntity]]] = None,
        sentiments: Optional[Dict[str, SentimentAnalysis]] = None,
        historical_accuracies: Optional[Dict[str, HistoricalAccuracy]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format multiple predictions for display.

        Requirement 4.4: Handle aggregated predictions with weighted confidence scores.
        """
        formatted_predictions = []

        for prediction in predictions:
            article = articles.get(prediction.article_id) if articles else None
            entity_list = entities.get(prediction.article_id) if entities else None
            sentiment = sentiments.get(prediction.article_id) if sentiments else None
            historical_accuracy = (
                historical_accuracies.get(prediction.stock_symbol)
                if historical_accuracies
                else None
            )

            formatted_prediction = self.format_prediction_display(
                prediction=prediction,
                article=article,
                entities=entity_list,
                sentiment=sentiment,
                historical_accuracy=historical_accuracy,
            )

            # Add aggregation metadata if this is an aggregated prediction
            if prediction.article_id.startswith("AGG_"):
                formatted_prediction["aggregation_info"] = {
                    "is_aggregated": True,
                    "aggregation_method": "weighted_confidence",
                    "note": "This prediction combines multiple articles for the same stock",
                }
            else:
                formatted_prediction["aggregation_info"] = {"is_aggregated": False}

            formatted_predictions.append(formatted_prediction)

        return formatted_predictions

    def export_to_json(
        self,
        predictions: List[MarketPrediction],
        articles: Optional[Dict[str, NewsArticle]] = None,
        entities: Optional[Dict[str, List[ExtractedEntity]]] = None,
        sentiments: Optional[Dict[str, SentimentAnalysis]] = None,
        historical_accuracies: Optional[Dict[str, HistoricalAccuracy]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Export predictions to JSON format.

        Requirement 4.5: Export results as structured JSON for external analysis.
        """
        formatted_data = self.format_aggregated_predictions(
            predictions=predictions,
            articles=articles,
            entities=entities,
            sentiments=sentiments,
            historical_accuracies=historical_accuracies,
        )

        # Add export metadata
        export_data = {
            "export_info": {
                "generated_at": datetime.now().isoformat(),
                "total_predictions": len(predictions),
                "format_version": "1.0",
            },
            "predictions": formatted_data,
        }

        json_content = json.dumps(export_data, indent=2, ensure_ascii=False)

        # Write to file if filename provided
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json_content)

        return json_content

    def export_to_csv(
        self,
        predictions: List[MarketPrediction],
        articles: Optional[Dict[str, NewsArticle]] = None,
        entities: Optional[Dict[str, List[ExtractedEntity]]] = None,
        sentiments: Optional[Dict[str, SentimentAnalysis]] = None,
        historical_accuracies: Optional[Dict[str, HistoricalAccuracy]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Export predictions to CSV format.

        Requirement 4.5: Export results as structured CSV for external analysis.
        """
        if not predictions:
            return ""

        # Flatten the data for CSV format
        csv_rows = []

        for prediction in predictions:
            article = articles.get(prediction.article_id) if articles else None
            entity_list = entities.get(prediction.article_id) if entities else None
            sentiment = sentiments.get(prediction.article_id) if sentiments else None
            historical_accuracy = (
                historical_accuracies.get(prediction.stock_symbol)
                if historical_accuracies
                else None
            )

            # Extract stock symbols and companies from entities
            stock_symbols = []
            companies = []
            financial_metrics = []

            if entity_list:
                stock_symbols = [
                    e.entity_value
                    for e in entity_list
                    if e.entity_type == "stock_symbol"
                ]
                companies = [
                    e.entity_value for e in entity_list if e.entity_type == "company"
                ]
                financial_metrics = [
                    e.entity_value for e in entity_list if e.entity_type == "metric"
                ]

            row = {
                # Core prediction data
                "prediction_id": prediction.article_id,
                "stock_symbol": prediction.stock_symbol,
                "impact_direction": prediction.impact_direction,
                "impact_magnitude": prediction.impact_magnitude,
                "confidence_level": prediction.confidence_level,
                "reasoning": prediction.reasoning,
                "created_at": prediction.created_at.isoformat(),
                # Article information
                "article_title": article.title if article else "",
                "article_published_at": (
                    article.published_at.isoformat() if article else ""
                ),
                "article_source": article.source if article else "",
                "article_category": article.category if article else "",
                "article_url": article.url if article else "",
                # Sentiment analysis
                "sentiment_score": sentiment.sentiment_score if sentiment else 0.0,
                "market_tone": sentiment.market_tone if sentiment else "neutral",
                "sentiment_confidence": sentiment.confidence if sentiment else 0.0,
                "key_phrases": "; ".join(sentiment.key_phrases) if sentiment else "",
                # Extracted entities (flattened)
                "mentioned_stock_symbols": "; ".join(stock_symbols),
                "mentioned_companies": "; ".join(companies),
                "financial_metrics": "; ".join(financial_metrics),
                # Historical accuracy
                "historical_accuracy_rate": (
                    historical_accuracy.accuracy_rate if historical_accuracy else 0.0
                ),
                "historical_total_predictions": (
                    historical_accuracy.total_predictions if historical_accuracy else 0
                ),
                "historical_time_period_days": (
                    historical_accuracy.time_period_days if historical_accuracy else 0
                ),
                # Aggregation info
                "is_aggregated": prediction.article_id.startswith("AGG_"),
            }

            csv_rows.append(row)

        # Create CSV content
        if not csv_rows:
            return ""

        output = StringIO()
        fieldnames = csv_rows[0].keys()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

        csv_content = output.getvalue()
        output.close()

        # Write to file if filename provided
        if filename:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                f.write(csv_content)

        return csv_content

    def create_summary_report(
        self,
        predictions: List[MarketPrediction],
        accuracy_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Create a summary report of all predictions.

        Provides high-level statistics and insights across all predictions.
        """
        if not predictions:
            return {"error": "No predictions to summarize"}

        # Basic statistics
        total_predictions = len(predictions)
        confidence_levels = [p.confidence_level for p in predictions]
        impact_magnitudes = [p.impact_magnitude for p in predictions]

        # Direction distribution
        directions = [p.impact_direction for p in predictions]
        direction_counts = {
            "positive": directions.count("positive"),
            "negative": directions.count("negative"),
            "neutral": directions.count("neutral"),
        }

        # Stock distribution
        stocks = [p.stock_symbol for p in predictions]
        unique_stocks = list(set(stocks))
        stock_counts = {stock: stocks.count(stock) for stock in unique_stocks}

        # Confidence distribution
        high_confidence = len([p for p in predictions if p.confidence_level >= 0.7])
        medium_confidence = len(
            [p for p in predictions if 0.3 <= p.confidence_level < 0.7]
        )
        low_confidence = len([p for p in predictions if p.confidence_level < 0.3])

        summary = {
            "report_generated_at": datetime.now().isoformat(),
            "total_predictions": total_predictions,
            "unique_stocks": len(unique_stocks),
            "statistics": {
                "average_confidence": sum(confidence_levels) / len(confidence_levels),
                "average_impact_magnitude": sum(impact_magnitudes)
                / len(impact_magnitudes),
                "max_confidence": max(confidence_levels),
                "min_confidence": min(confidence_levels),
            },
            "direction_distribution": direction_counts,
            "confidence_distribution": {
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": low_confidence,
            },
            "stock_distribution": stock_counts,
            "top_stocks_by_prediction_count": sorted(
                stock_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

        # Add accuracy metrics if provided
        if accuracy_metrics:
            summary["accuracy_metrics"] = accuracy_metrics

        return summary
