"""
Core data models for the News Market Predictor system.
"""

import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass, asdict
from io import StringIO


class ValidationError(Exception):
    """Raised when data validation fails."""


@dataclass
class NewsArticle:
    """Structured representation of a Yahoo Finance news item."""

    id: str
    title: str
    content: str
    url: str
    published_at: datetime
    source: str
    category: str
    raw_metadata: Dict

    def validate(self) -> bool:
        """Validate NewsArticle data with proper type checking."""
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValidationError("Article ID must be a non-empty string")

        if not isinstance(self.title, str) or not self.title.strip():
            raise ValidationError("Article title must be a non-empty string")

        if not isinstance(self.content, str) or not self.content.strip():
            raise ValidationError("Article content must be a non-empty string")

        if not isinstance(self.url, str) or not self.url.strip():
            raise ValidationError("Article URL must be a non-empty string")

        if not isinstance(self.published_at, datetime):
            raise ValidationError("Published date must be a datetime object")

        if not isinstance(self.source, str) or not self.source.strip():
            raise ValidationError("Article source must be a non-empty string")

        if not isinstance(self.category, str) or not self.category.strip():
            raise ValidationError("Article category must be a non-empty string")

        if not isinstance(self.raw_metadata, dict):
            raise ValidationError("Raw metadata must be a dictionary")

        return True

    def to_json(self) -> str:
        """Serialize NewsArticle to JSON format."""
        self.validate()
        data = asdict(self)
        data["published_at"] = self.published_at.isoformat()
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "NewsArticle":
        """Deserialize NewsArticle from JSON format."""
        try:
            data = json.loads(json_str)
            data["published_at"] = datetime.fromisoformat(data["published_at"])
            article = cls(**data)
            article.validate()
            return article
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid JSON format for NewsArticle: {e}") from e

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert NewsArticle to CSV row format."""
        self.validate()
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "source": self.source,
            "category": self.category,
            "raw_metadata": json.dumps(self.raw_metadata),
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "NewsArticle":
        """Create NewsArticle from CSV row format."""
        try:
            row["published_at"] = datetime.fromisoformat(row["published_at"])
            row["raw_metadata"] = json.loads(row["raw_metadata"])
            article = cls(**row)
            article.validate()
            return article
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            raise ValidationError(f"Invalid CSV row format for NewsArticle: {e}") from e


@dataclass
class SentimentAnalysis:
    """Results of sentiment analysis on a news article."""

    article_id: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    key_phrases: List[str]
    market_tone: str  # bullish, bearish, neutral

    def validate(self) -> bool:
        """Validate SentimentAnalysis data with proper type checking."""
        if not isinstance(self.article_id, str) or not self.article_id.strip():
            raise ValidationError("Article ID must be a non-empty string")

        if not isinstance(self.sentiment_score, (int, float)):
            raise ValidationError("Sentiment score must be a number")

        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValidationError("Sentiment score must be between -1.0 and 1.0")

        if not isinstance(self.confidence, (int, float)):
            raise ValidationError("Confidence must be a number")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("Confidence must be between 0.0 and 1.0")

        if not isinstance(self.key_phrases, list):
            raise ValidationError("Key phrases must be a list")

        if not all(isinstance(phrase, str) for phrase in self.key_phrases):
            raise ValidationError("All key phrases must be strings")

        if self.market_tone not in ["bullish", "bearish", "neutral"]:
            raise ValidationError(
                "Market tone must be 'bullish', 'bearish', or 'neutral'"
            )

        return True

    def to_json(self) -> str:
        """Serialize SentimentAnalysis to JSON format."""
        self.validate()
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SentimentAnalysis":
        """Deserialize SentimentAnalysis from JSON format."""
        try:
            data = json.loads(json_str)
            sentiment = cls(**data)
            sentiment.validate()
            return sentiment
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid JSON format for SentimentAnalysis: {e}"
            ) from e

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert SentimentAnalysis to CSV row format."""
        self.validate()
        return {
            "article_id": self.article_id,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "key_phrases": json.dumps(self.key_phrases),
            "market_tone": self.market_tone,
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "SentimentAnalysis":
        """Create SentimentAnalysis from CSV row format."""
        try:
            row["key_phrases"] = json.loads(row["key_phrases"])
            row["sentiment_score"] = float(row["sentiment_score"])
            row["confidence"] = float(row["confidence"])
            sentiment = cls(**row)
            sentiment.validate()
            return sentiment
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            raise ValidationError(
                f"Invalid CSV row format for SentimentAnalysis: {e}"
            ) from e


@dataclass
class ExtractedEntity:
    """Entity extracted from news article content."""

    article_id: str
    entity_type: str  # stock_symbol, company, metric
    entity_value: str
    relevance_score: float  # 0.0 to 1.0
    context: str

    def validate(self) -> bool:
        """Validate ExtractedEntity data with proper type checking."""
        if not isinstance(self.article_id, str) or not self.article_id.strip():
            raise ValidationError("Article ID must be a non-empty string")

        if self.entity_type not in ["stock_symbol", "company", "metric"]:
            raise ValidationError(
                "Entity type must be 'stock_symbol', 'company', or 'metric'"
            )

        if not isinstance(self.entity_value, str) or not self.entity_value.strip():
            raise ValidationError("Entity value must be a non-empty string")

        if not isinstance(self.relevance_score, (int, float)):
            raise ValidationError("Relevance score must be a number")

        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValidationError("Relevance score must be between 0.0 and 1.0")

        if not isinstance(self.context, str):
            raise ValidationError("Context must be a string")

        return True

    def to_json(self) -> str:
        """Serialize ExtractedEntity to JSON format."""
        self.validate()
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ExtractedEntity":
        """Deserialize ExtractedEntity from JSON format."""
        try:
            data = json.loads(json_str)
            entity = cls(**data)
            entity.validate()
            return entity
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid JSON format for ExtractedEntity: {e}"
            ) from e

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert ExtractedEntity to CSV row format."""
        self.validate()
        return asdict(self)

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "ExtractedEntity":
        """Create ExtractedEntity from CSV row format."""
        try:
            row["relevance_score"] = float(row["relevance_score"])
            entity = cls(**row)
            entity.validate()
            return entity
        except (KeyError, ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid CSV row format for ExtractedEntity: {e}"
            ) from e


@dataclass
class MarketPrediction:
    """Prediction of market impact from a news article."""

    article_id: str
    stock_symbol: str
    impact_direction: str  # positive, negative, neutral
    impact_magnitude: float  # 0.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    reasoning: str
    created_at: datetime

    def validate(self) -> bool:
        """Validate MarketPrediction data with proper type checking."""
        if not isinstance(self.article_id, str) or not self.article_id.strip():
            raise ValidationError("Article ID must be a non-empty string")

        if not isinstance(self.stock_symbol, str) or not self.stock_symbol.strip():
            raise ValidationError("Stock symbol must be a non-empty string")

        if self.impact_direction not in ["positive", "negative", "neutral"]:
            raise ValidationError(
                "Impact direction must be 'positive', 'negative', or 'neutral'"
            )

        if not isinstance(self.impact_magnitude, (int, float)):
            raise ValidationError("Impact magnitude must be a number")

        if not 0.0 <= self.impact_magnitude <= 1.0:
            raise ValidationError("Impact magnitude must be between 0.0 and 1.0")

        if not isinstance(self.confidence_level, (int, float)):
            raise ValidationError("Confidence level must be a number")

        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValidationError("Confidence level must be between 0.0 and 1.0")

        if not isinstance(self.reasoning, str):
            raise ValidationError("Reasoning must be a string")

        if not isinstance(self.created_at, datetime):
            raise ValidationError("Created date must be a datetime object")

        return True

    def to_json(self) -> str:
        """Serialize MarketPrediction to JSON format."""
        self.validate()
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MarketPrediction":
        """Deserialize MarketPrediction from JSON format."""
        try:
            data = json.loads(json_str)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            prediction = cls(**data)
            prediction.validate()
            return prediction
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid JSON format for MarketPrediction: {e}"
            ) from e

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert MarketPrediction to CSV row format."""
        self.validate()
        return {
            "article_id": self.article_id,
            "stock_symbol": self.stock_symbol,
            "impact_direction": self.impact_direction,
            "impact_magnitude": self.impact_magnitude,
            "confidence_level": self.confidence_level,
            "reasoning": self.reasoning,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "MarketPrediction":
        """Create MarketPrediction from CSV row format."""
        try:
            row["created_at"] = datetime.fromisoformat(row["created_at"])
            row["impact_magnitude"] = float(row["impact_magnitude"])
            row["confidence_level"] = float(row["confidence_level"])
            prediction = cls(**row)
            prediction.validate()
            return prediction
        except (KeyError, ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid CSV row format for MarketPrediction: {e}"
            ) from e


@dataclass
class MarketOutcome:
    """Actual market outcome for a stock after a prediction."""

    prediction_id: str
    stock_symbol: str
    actual_direction: str  # positive, negative, neutral
    actual_magnitude: float  # 0.0 to 1.0
    price_change_percent: float  # actual percentage change
    evaluation_date: datetime  # when the outcome was measured
    time_horizon_hours: int  # hours after prediction when measured

    def validate(self) -> bool:
        """Validate MarketOutcome data with proper type checking."""
        if not isinstance(self.prediction_id, str) or not self.prediction_id.strip():
            raise ValidationError("Prediction ID must be a non-empty string")

        if not isinstance(self.stock_symbol, str) or not self.stock_symbol.strip():
            raise ValidationError("Stock symbol must be a non-empty string")

        if self.actual_direction not in ["positive", "negative", "neutral"]:
            raise ValidationError(
                "Actual direction must be 'positive', 'negative', or 'neutral'"
            )

        if not isinstance(self.actual_magnitude, (int, float)):
            raise ValidationError("Actual magnitude must be a number")

        if not 0.0 <= self.actual_magnitude <= 1.0:
            raise ValidationError("Actual magnitude must be between 0.0 and 1.0")

        if not isinstance(self.price_change_percent, (int, float)):
            raise ValidationError("Price change percent must be a number")

        if not isinstance(self.evaluation_date, datetime):
            raise ValidationError("Evaluation date must be a datetime object")

        if not isinstance(self.time_horizon_hours, int) or self.time_horizon_hours <= 0:
            raise ValidationError("Time horizon hours must be a positive integer")

        return True

    def to_json(self) -> str:
        """Serialize MarketOutcome to JSON format."""
        self.validate()
        data = asdict(self)
        data["evaluation_date"] = self.evaluation_date.isoformat()
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MarketOutcome":
        """Deserialize MarketOutcome from JSON format."""
        try:
            data = json.loads(json_str)
            data["evaluation_date"] = datetime.fromisoformat(data["evaluation_date"])
            outcome = cls(**data)
            outcome.validate()
            return outcome
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid JSON format for MarketOutcome: {e}") from e

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert MarketOutcome to CSV row format."""
        self.validate()
        return {
            "prediction_id": self.prediction_id,
            "stock_symbol": self.stock_symbol,
            "actual_direction": self.actual_direction,
            "actual_magnitude": self.actual_magnitude,
            "price_change_percent": self.price_change_percent,
            "evaluation_date": self.evaluation_date.isoformat(),
            "time_horizon_hours": self.time_horizon_hours,
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "MarketOutcome":
        """Create MarketOutcome from CSV row format."""
        try:
            row["evaluation_date"] = datetime.fromisoformat(row["evaluation_date"])
            row["actual_magnitude"] = float(row["actual_magnitude"])
            row["price_change_percent"] = float(row["price_change_percent"])
            row["time_horizon_hours"] = int(row["time_horizon_hours"])
            outcome = cls(**row)
            outcome.validate()
            return outcome
        except (KeyError, ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid CSV row format for MarketOutcome: {e}"
            ) from e


@dataclass
class HistoricalAccuracy:
    """Historical accuracy metrics for predictions."""

    stock_symbol: str
    time_period_days: int
    total_predictions: int
    correct_predictions: int
    accuracy_rate: float  # 0.0 to 1.0
    average_confidence: float  # 0.0 to 1.0
    calculated_at: datetime

    def validate(self) -> bool:
        """Validate HistoricalAccuracy data with proper type checking."""
        if not isinstance(self.stock_symbol, str) or not self.stock_symbol.strip():
            raise ValidationError("Stock symbol must be a non-empty string")

        if not isinstance(self.time_period_days, int) or self.time_period_days <= 0:
            raise ValidationError("Time period days must be a positive integer")

        if not isinstance(self.total_predictions, int) or self.total_predictions < 0:
            raise ValidationError("Total predictions must be a non-negative integer")

        if (
            not isinstance(self.correct_predictions, int)
            or self.correct_predictions < 0
        ):
            raise ValidationError("Correct predictions must be a non-negative integer")

        if self.correct_predictions > self.total_predictions:
            raise ValidationError("Correct predictions cannot exceed total predictions")

        if not isinstance(self.accuracy_rate, (int, float)):
            raise ValidationError("Accuracy rate must be a number")

        if not 0.0 <= self.accuracy_rate <= 1.0:
            raise ValidationError("Accuracy rate must be between 0.0 and 1.0")

        if not isinstance(self.average_confidence, (int, float)):
            raise ValidationError("Average confidence must be a number")

        if not 0.0 <= self.average_confidence <= 1.0:
            raise ValidationError("Average confidence must be between 0.0 and 1.0")

        if not isinstance(self.calculated_at, datetime):
            raise ValidationError("Calculated date must be a datetime object")

        return True

    def to_json(self) -> str:
        """Serialize HistoricalAccuracy to JSON format."""
        self.validate()
        data = asdict(self)
        data["calculated_at"] = self.calculated_at.isoformat()
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "HistoricalAccuracy":
        """Deserialize HistoricalAccuracy from JSON format."""
        try:
            data = json.loads(json_str)
            data["calculated_at"] = datetime.fromisoformat(data["calculated_at"])
            accuracy = cls(**data)
            accuracy.validate()
            return accuracy
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid JSON format for HistoricalAccuracy: {e}"
            ) from e

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert HistoricalAccuracy to CSV row format."""
        self.validate()
        return {
            "stock_symbol": self.stock_symbol,
            "time_period_days": self.time_period_days,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy_rate": self.accuracy_rate,
            "average_confidence": self.average_confidence,
            "calculated_at": self.calculated_at.isoformat(),
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "HistoricalAccuracy":
        """Create HistoricalAccuracy from CSV row format."""
        try:
            row["calculated_at"] = datetime.fromisoformat(row["calculated_at"])
            row["time_period_days"] = int(row["time_period_days"])
            row["total_predictions"] = int(row["total_predictions"])
            row["correct_predictions"] = int(row["correct_predictions"])
            row["accuracy_rate"] = float(row["accuracy_rate"])
            row["average_confidence"] = float(row["average_confidence"])
            accuracy = cls(**row)
            accuracy.validate()
            return accuracy
        except (KeyError, ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid CSV row format for HistoricalAccuracy: {e}"
            ) from e


def export_to_csv(
    data_objects: List[
        Union[
            NewsArticle,
            SentimentAnalysis,
            ExtractedEntity,
            MarketPrediction,
            MarketOutcome,
            HistoricalAccuracy,
        ]
    ],
    filename: str = None,
) -> str:
    """Export a list of data objects to CSV format."""
    if not data_objects:
        raise ValidationError("Cannot export empty list to CSV")

    # Get the type of objects and validate they're all the same type
    obj_type = type(data_objects[0])
    if not all(isinstance(obj, obj_type) for obj in data_objects):
        raise ValidationError("All objects must be of the same type for CSV export")

    # Convert to CSV rows
    csv_rows = [obj.to_csv_row() for obj in data_objects]

    # Create CSV string
    output = StringIO()
    if csv_rows:
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


def export_to_json(
    data_objects: List[
        Union[
            NewsArticle,
            SentimentAnalysis,
            ExtractedEntity,
            MarketPrediction,
            MarketOutcome,
            HistoricalAccuracy,
        ]
    ],
    filename: str = None,
) -> str:
    """Export a list of data objects to JSON format."""
    if not data_objects:
        raise ValidationError("Cannot export empty list to JSON")

    # Convert to JSON-serializable format
    json_data = []
    for obj in data_objects:
        obj.validate()
        data = asdict(obj)
        # Handle datetime serialization
        if hasattr(obj, "published_at") and isinstance(obj.published_at, datetime):
            data["published_at"] = obj.published_at.isoformat()
        if hasattr(obj, "created_at") and isinstance(obj.created_at, datetime):
            data["created_at"] = obj.created_at.isoformat()
        if hasattr(obj, "evaluation_date") and isinstance(
            obj.evaluation_date, datetime
        ):
            data["evaluation_date"] = obj.evaluation_date.isoformat()
        if hasattr(obj, "calculated_at") and isinstance(obj.calculated_at, datetime):
            data["calculated_at"] = obj.calculated_at.isoformat()
        json_data.append(data)

    json_content = json.dumps(json_data, indent=2)

    # Write to file if filename provided
    if filename:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json_content)

    return json_content
