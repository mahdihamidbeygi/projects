"""
Data Access Layer (DAL) for the News Market Predictor system.
Provides a clean interface for all database operations with proper error handling.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from ..models import (
    NewsArticle,
    SentimentAnalysis,
    ExtractedEntity,
    MarketPrediction,
    MarketOutcome,
    HistoricalAccuracy,
    ValidationError,
)
from ..interfaces import DataStorage, HistoricalDataInterface
from ..exceptions import StorageError
from .database_connection import DatabaseConnection

logger = logging.getLogger(__name__)


class DataAccessLayer(DataStorage, HistoricalDataInterface):
    """
    Data Access Layer providing clean interface to database operations.
    Implements both DataStorage and HistoricalDataInterface.
    """

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize data access layer."""
        self.db = db_connection

    # Article operations

    def store_article(self, article: NewsArticle) -> bool:
        """Store a news article in the database."""
        try:
            article.validate()

            query = """
                INSERT OR REPLACE INTO articles 
                (id, title, content, url, published_at, source, category, raw_metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = (
                article.id,
                article.title,
                article.content,
                article.url,
                article.published_at.isoformat(),
                article.source,
                article.category,
                json.dumps(article.raw_metadata),
                datetime.now().isoformat(),
            )

            self.db.execute_update(query, params)
            logger.debug(f"Stored article {article.id}")
            return True

        except (ValidationError, Exception) as e:
            logger.error(f"Failed to store article: {e}")
            raise StorageError(f"Failed to store article: {e}") from e

    def retrieve_articles(
        self, date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[NewsArticle]:
        """Retrieve articles within date range."""
        try:
            query = "SELECT * FROM articles"
            params = []

            if date_range:
                query += " WHERE published_at BETWEEN ? AND ?"
                params = [date_range[0].isoformat(), date_range[1].isoformat()]

            query += " ORDER BY published_at DESC"

            rows = self.db.execute_query(query, tuple(params))

            articles = []
            for row in rows:
                article = NewsArticle(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    url=row[3],
                    published_at=datetime.fromisoformat(row[4]),
                    source=row[5],
                    category=row[6],
                    raw_metadata=json.loads(row[7]),
                )
                articles.append(article)

            return articles

        except Exception as e:
            logger.error(f"Failed to retrieve articles: {e}")
            return []

    def get_article_by_id(self, article_id: str) -> Optional[NewsArticle]:
        """Get a specific article by ID."""
        try:
            query = "SELECT * FROM articles WHERE id = ?"
            row = self.db.execute_query(query, (article_id,), fetch_one=True)

            if row:
                return NewsArticle(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    url=row[3],
                    published_at=datetime.fromisoformat(row[4]),
                    source=row[5],
                    category=row[6],
                    raw_metadata=json.loads(row[7]),
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get article {article_id}: {e}")
            return None

    # Sentiment analysis operations

    def store_sentiment(self, sentiment: SentimentAnalysis) -> bool:
        """Store sentiment analysis results."""
        try:
            sentiment.validate()

            query = """
                INSERT OR REPLACE INTO sentiment_analysis 
                (article_id, sentiment_score, confidence, key_phrases, market_tone, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """

            params = (
                sentiment.article_id,
                sentiment.sentiment_score,
                sentiment.confidence,
                json.dumps(sentiment.key_phrases),
                sentiment.market_tone,
                datetime.now().isoformat(),
            )

            self.db.execute_update(query, params)
            logger.debug(f"Stored sentiment for article {sentiment.article_id}")
            return True

        except (ValidationError, Exception) as e:
            logger.error(f"Failed to store sentiment: {e}")
            raise StorageError(f"Failed to store sentiment: {e}") from e

    def get_sentiment_by_article(self, article_id: str) -> Optional[SentimentAnalysis]:
        """Get sentiment analysis for an article."""
        try:
            query = "SELECT * FROM sentiment_analysis WHERE article_id = ?"
            row = self.db.execute_query(query, (article_id,), fetch_one=True)

            if row:
                return SentimentAnalysis(
                    article_id=row[1],
                    sentiment_score=row[2],
                    confidence=row[3],
                    key_phrases=json.loads(row[4]),
                    market_tone=row[5],
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get sentiment for article {article_id}: {e}")
            return None

    # Entity operations

    def store_entities(self, entities: List[ExtractedEntity]) -> bool:
        """Store extracted entities."""
        try:
            if not entities:
                return True

            # Validate all entities first
            for entity in entities:
                entity.validate()

            query = """
                INSERT INTO extracted_entities 
                (article_id, entity_type, entity_value, relevance_score, context, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """

            params_list = [
                (
                    entity.article_id,
                    entity.entity_type,
                    entity.entity_value,
                    entity.relevance_score,
                    entity.context,
                    datetime.now().isoformat(),
                )
                for entity in entities
            ]

            self.db.execute_many(query, params_list)
            logger.debug(f"Stored {len(entities)} entities")
            return True

        except (ValidationError, Exception) as e:
            logger.error(f"Failed to store entities: {e}")
            raise StorageError(f"Failed to store entities: {e}") from e

    def get_entities_by_article(self, article_id: str) -> List[ExtractedEntity]:
        """Get all entities for an article."""
        try:
            query = "SELECT * FROM extracted_entities WHERE article_id = ?"
            rows = self.db.execute_query(query, (article_id,))

            entities = []
            for row in rows:
                entity = ExtractedEntity(
                    article_id=row[1],
                    entity_type=row[2],
                    entity_value=row[3],
                    relevance_score=row[4],
                    context=row[5],
                )
                entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"Failed to get entities for article {article_id}: {e}")
            return []

    # Prediction operations

    def store_prediction(self, prediction: MarketPrediction) -> bool:
        """Store a market prediction."""
        try:
            prediction.validate()

            query = """
                INSERT OR REPLACE INTO predictions 
                (article_id, stock_symbol, impact_direction, impact_magnitude, 
                 confidence_level, reasoning, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            params = (
                prediction.article_id,
                prediction.stock_symbol,
                prediction.impact_direction,
                prediction.impact_magnitude,
                prediction.confidence_level,
                prediction.reasoning,
                prediction.created_at.isoformat(),
            )

            self.db.execute_update(query, params)
            logger.debug(
                f"Stored prediction for {prediction.stock_symbol} from article {prediction.article_id}"
            )
            return True

        except (ValidationError, Exception) as e:
            logger.error(f"Failed to store prediction: {e}")
            raise StorageError(f"Failed to store prediction: {e}") from e

    def retrieve_predictions(
        self, stock_symbol: Optional[str] = None
    ) -> List[MarketPrediction]:
        """Retrieve predictions for a specific stock or all stocks."""
        try:
            query = "SELECT * FROM predictions"
            params = []

            if stock_symbol:
                query += " WHERE stock_symbol = ?"
                params.append(stock_symbol)

            query += " ORDER BY created_at DESC"

            rows = self.db.execute_query(query, tuple(params))

            predictions = []
            for row in rows:
                prediction = MarketPrediction(
                    article_id=row[1],
                    stock_symbol=row[2],
                    impact_direction=row[3],
                    impact_magnitude=row[4],
                    confidence_level=row[5],
                    reasoning=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                )
                predictions.append(prediction)

            return predictions

        except Exception as e:
            logger.error(f"Failed to retrieve predictions: {e}")
            return []

    # Outcome operations

    def store_outcome(self, outcome: MarketOutcome) -> bool:
        """Store a market outcome."""
        try:
            outcome.validate()

            query = """
                INSERT OR REPLACE INTO outcomes 
                (prediction_id, stock_symbol, actual_direction, actual_magnitude,
                 price_change_percent, evaluation_date, time_horizon_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            params = (
                outcome.prediction_id,
                outcome.stock_symbol,
                outcome.actual_direction,
                outcome.actual_magnitude,
                outcome.price_change_percent,
                outcome.evaluation_date.isoformat(),
                outcome.time_horizon_hours,
            )

            self.db.execute_update(query, params)
            logger.debug(f"Stored outcome for prediction {outcome.prediction_id}")
            return True

        except (ValidationError, Exception) as e:
            logger.error(f"Failed to store outcome: {e}")
            raise StorageError(f"Failed to store outcome: {e}") from e

    def get_outcomes_by_prediction(self, prediction_id: str) -> List[MarketOutcome]:
        """Get all outcomes for a prediction."""
        try:
            query = "SELECT * FROM outcomes WHERE prediction_id = ?"
            rows = self.db.execute_query(query, (prediction_id,))

            outcomes = []
            for row in rows:
                outcome = MarketOutcome(
                    prediction_id=row[1],
                    stock_symbol=row[2],
                    actual_direction=row[3],
                    actual_magnitude=row[4],
                    price_change_percent=row[5],
                    evaluation_date=datetime.fromisoformat(row[6]),
                    time_horizon_hours=row[7],
                )
                outcomes.append(outcome)

            return outcomes

        except Exception as e:
            logger.error(f"Failed to get outcomes for prediction {prediction_id}: {e}")
            return []

    # Accuracy metrics operations

    def store_accuracy_metrics(self, accuracy: HistoricalAccuracy) -> bool:
        """Store historical accuracy metrics."""
        try:
            accuracy.validate()

            query = """
                INSERT OR REPLACE INTO accuracy_metrics 
                (stock_symbol, time_period_days, total_predictions, correct_predictions,
                 accuracy_rate, average_confidence, calculated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            params = (
                accuracy.stock_symbol,
                accuracy.time_period_days,
                accuracy.total_predictions,
                accuracy.correct_predictions,
                accuracy.accuracy_rate,
                accuracy.average_confidence,
                accuracy.calculated_at.isoformat(),
            )

            self.db.execute_update(query, params)
            logger.debug(f"Stored accuracy metrics for {accuracy.stock_symbol}")
            return True

        except (ValidationError, Exception) as e:
            logger.error(f"Failed to store accuracy metrics: {e}")
            raise StorageError(f"Failed to store accuracy metrics: {e}") from e

    def calculate_historical_accuracy(
        self, stock_symbol: str, time_period_days: int = 30
    ) -> Optional[HistoricalAccuracy]:
        """Calculate historical accuracy for a stock over a time period."""
        try:
            from datetime import timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_period_days)

            # Get predictions in the time period
            query = """
                SELECT * FROM predictions 
                WHERE stock_symbol = ? AND created_at BETWEEN ? AND ?
            """
            rows = self.db.execute_query(
                query, (stock_symbol, start_date.isoformat(), end_date.isoformat())
            )

            if not rows:
                return None

            predictions = []
            for row in rows:
                prediction = MarketPrediction(
                    article_id=row[1],
                    stock_symbol=row[2],
                    impact_direction=row[3],
                    impact_magnitude=row[4],
                    confidence_level=row[5],
                    reasoning=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                )
                predictions.append(prediction)

            # Calculate accuracy
            correct_predictions = 0
            total_confidence = 0.0

            for prediction in predictions:
                outcomes = self.get_outcomes_by_prediction(prediction.article_id)
                if outcomes:
                    outcome = outcomes[0]
                    if prediction.impact_direction == outcome.actual_direction:
                        correct_predictions += 1

                total_confidence += prediction.confidence_level

            total_predictions = len(predictions)
            accuracy_rate = (
                correct_predictions / total_predictions
                if total_predictions > 0
                else 0.0
            )
            average_confidence = (
                total_confidence / total_predictions if total_predictions > 0 else 0.0
            )

            accuracy = HistoricalAccuracy(
                stock_symbol=stock_symbol,
                time_period_days=time_period_days,
                total_predictions=total_predictions,
                correct_predictions=correct_predictions,
                accuracy_rate=accuracy_rate,
                average_confidence=average_confidence,
                calculated_at=datetime.now(),
            )

            # Store the calculated accuracy
            self.store_accuracy_metrics(accuracy)

            return accuracy

        except Exception as e:
            logger.error(f"Failed to calculate historical accuracy: {e}")
            return None

    def get_similar_historical_predictions(
        self, stock_symbol: str, sentiment_score: float, lookback_days: int = 90
    ) -> List[Tuple[MarketPrediction, Optional[MarketOutcome]]]:
        """Get similar historical predictions for influence calculation."""
        try:
            from datetime import timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # Get predictions for the same stock in the lookback period
            query = """
                SELECT * FROM predictions 
                WHERE stock_symbol = ? AND created_at BETWEEN ? AND ?
            """
            rows = self.db.execute_query(
                query, (stock_symbol, start_date.isoformat(), end_date.isoformat())
            )

            similar_predictions = []
            for row in rows:
                prediction = MarketPrediction(
                    article_id=row[1],
                    stock_symbol=row[2],
                    impact_direction=row[3],
                    impact_magnitude=row[4],
                    confidence_level=row[5],
                    reasoning=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                )

                outcomes = self.get_outcomes_by_prediction(prediction.article_id)
                outcome = outcomes[0] if outcomes else None
                similar_predictions.append((prediction, outcome))

            return similar_predictions

        except Exception as e:
            logger.error(f"Failed to get similar historical predictions: {e}")
            return []

    def cleanup_old_data(self, retention_days: int = 365) -> bool:
        """Clean up old data based on retention policy."""
        try:
            from datetime import timedelta

            cutoff_date = datetime.now() - timedelta(days=retention_days)

            # Delete old articles and related data
            with self.db.transaction() as conn:
                cursor = conn.cursor()

                # Delete old predictions
                cursor.execute(
                    "DELETE FROM predictions WHERE created_at < ?",
                    (cutoff_date.isoformat(),),
                )

                # Delete old outcomes
                cursor.execute(
                    "DELETE FROM outcomes WHERE evaluation_date < ?",
                    (cutoff_date.isoformat(),),
                )

                # Delete old accuracy metrics (keep more recent ones)
                accuracy_cutoff = datetime.now() - timedelta(days=90)
                cursor.execute(
                    "DELETE FROM accuracy_metrics WHERE calculated_at < ?",
                    (accuracy_cutoff.isoformat(),),
                )

                # Delete old articles
                cursor.execute(
                    "DELETE FROM articles WHERE published_at < ?",
                    (cutoff_date.isoformat(),),
                )

                # Delete orphaned sentiment analysis
                cursor.execute(
                    """
                    DELETE FROM sentiment_analysis 
                    WHERE article_id NOT IN (SELECT id FROM articles)
                """
                )

                # Delete orphaned entities
                cursor.execute(
                    """
                    DELETE FROM extracted_entities 
                    WHERE article_id NOT IN (SELECT id FROM articles)
                """
                )

            logger.info(f"Cleaned up data older than {retention_days} days")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False

    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about the database contents."""
        try:
            stats = {}

            # Count articles
            result = self.db.execute_query(
                "SELECT COUNT(*) FROM articles", fetch_one=True
            )
            stats["total_articles"] = result[0] if result else 0

            # Count predictions
            result = self.db.execute_query(
                "SELECT COUNT(*) FROM predictions", fetch_one=True
            )
            stats["total_predictions"] = result[0] if result else 0

            # Count outcomes
            result = self.db.execute_query(
                "SELECT COUNT(*) FROM outcomes", fetch_one=True
            )
            stats["total_outcomes"] = result[0] if result else 0

            # Count accuracy metrics
            result = self.db.execute_query(
                "SELECT COUNT(*) FROM accuracy_metrics", fetch_one=True
            )
            stats["total_accuracy_records"] = result[0] if result else 0

            # Count unique stocks
            result = self.db.execute_query(
                "SELECT COUNT(DISTINCT stock_symbol) FROM predictions", fetch_one=True
            )
            stats["unique_stocks"] = result[0] if result else 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
