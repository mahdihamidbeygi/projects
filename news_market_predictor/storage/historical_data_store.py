"""
Historical data storage and management for the News Market Predictor system.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from ..models import (
    MarketPrediction,
    MarketOutcome,
    HistoricalAccuracy,
    ValidationError,
)
from ..interfaces import DataStorage


class HistoricalDataStore(DataStorage):
    """SQLite-based storage for historical predictions and outcomes."""

    def __init__(self, db_path: str = "historical_data.db"):
        """Initialize the historical data store with database connection."""
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema for historical data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create predictions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    impact_direction TEXT NOT NULL,
                    impact_magnitude REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(article_id, stock_symbol)
                )
            """
            )

            # Create outcomes table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    actual_direction TEXT NOT NULL,
                    actual_magnitude REAL NOT NULL,
                    price_change_percent REAL NOT NULL,
                    evaluation_date TEXT NOT NULL,
                    time_horizon_hours INTEGER NOT NULL,
                    UNIQUE(prediction_id, time_horizon_hours)
                )
            """
            )

            # Create accuracy metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS accuracy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_symbol TEXT NOT NULL,
                    time_period_days INTEGER NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    accuracy_rate REAL NOT NULL,
                    average_confidence REAL NOT NULL,
                    calculated_at TEXT NOT NULL,
                    UNIQUE(stock_symbol, time_period_days, calculated_at)
                )
            """
            )

            # Create indexes for better query performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_stock_date ON predictions(stock_symbol, created_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_outcomes_prediction ON outcomes(prediction_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_accuracy_stock ON accuracy_metrics(stock_symbol)"
            )

            conn.commit()

    def store_article(self, article) -> bool:
        """Store a news article (not implemented in historical store)."""
        raise NotImplementedError("Historical data store does not handle articles")

    def store_prediction(self, prediction: MarketPrediction) -> bool:
        """Store a market prediction in the database."""
        try:
            prediction.validate()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO predictions 
                    (article_id, stock_symbol, impact_direction, impact_magnitude, 
                     confidence_level, reasoning, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        prediction.article_id,
                        prediction.stock_symbol,
                        prediction.impact_direction,
                        prediction.impact_magnitude,
                        prediction.confidence_level,
                        prediction.reasoning,
                        prediction.created_at.isoformat(),
                    ),
                )
                conn.commit()
                return True

        except (sqlite3.Error, ValidationError) as e:
            print(f"Error storing prediction: {e}")
            return False

    def store_outcome(self, outcome: MarketOutcome) -> bool:
        """Store a market outcome in the database."""
        try:
            outcome.validate()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO outcomes 
                    (prediction_id, stock_symbol, actual_direction, actual_magnitude,
                     price_change_percent, evaluation_date, time_horizon_hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        outcome.prediction_id,
                        outcome.stock_symbol,
                        outcome.actual_direction,
                        outcome.actual_magnitude,
                        outcome.price_change_percent,
                        outcome.evaluation_date.isoformat(),
                        outcome.time_horizon_hours,
                    ),
                )
                conn.commit()
                return True

        except (sqlite3.Error, ValidationError) as e:
            print(f"Error storing outcome: {e}")
            return False

    def store_accuracy_metrics(self, accuracy: HistoricalAccuracy) -> bool:
        """Store historical accuracy metrics in the database."""
        try:
            accuracy.validate()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO accuracy_metrics 
                    (stock_symbol, time_period_days, total_predictions, correct_predictions,
                     accuracy_rate, average_confidence, calculated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        accuracy.stock_symbol,
                        accuracy.time_period_days,
                        accuracy.total_predictions,
                        accuracy.correct_predictions,
                        accuracy.accuracy_rate,
                        accuracy.average_confidence,
                        accuracy.calculated_at.isoformat(),
                    ),
                )
                conn.commit()
                return True

        except (sqlite3.Error, ValidationError) as e:
            print(f"Error storing accuracy metrics: {e}")
            return False

    def retrieve_articles(self, date_range: Optional[tuple] = None) -> List:
        """Retrieve articles (not implemented in historical store)."""
        raise NotImplementedError("Historical data store does not handle articles")

    def retrieve_predictions(
        self,
        stock_symbol: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[MarketPrediction]:
        """Retrieve predictions for a specific stock or all stocks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM predictions"
                params = []
                conditions = []

                if stock_symbol:
                    conditions.append("stock_symbol = ?")
                    params.append(stock_symbol)

                if date_range:
                    conditions.append("created_at BETWEEN ? AND ?")
                    params.extend(
                        [date_range[0].isoformat(), date_range[1].isoformat()]
                    )

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY created_at DESC"

                cursor.execute(query, params)
                rows = cursor.fetchall()

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

        except sqlite3.Error as e:
            print(f"Error retrieving predictions: {e}")
            return []

    def retrieve_outcomes(
        self, prediction_id: Optional[str] = None, stock_symbol: Optional[str] = None
    ) -> List[MarketOutcome]:
        """Retrieve market outcomes for predictions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM outcomes"
                params = []
                conditions = []

                if prediction_id:
                    conditions.append("prediction_id = ?")
                    params.append(prediction_id)

                if stock_symbol:
                    conditions.append("stock_symbol = ?")
                    params.append(stock_symbol)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY evaluation_date DESC"

                cursor.execute(query, params)
                rows = cursor.fetchall()

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

        except sqlite3.Error as e:
            print(f"Error retrieving outcomes: {e}")
            return []

    def retrieve_accuracy_metrics(
        self, stock_symbol: Optional[str] = None, time_period_days: Optional[int] = None
    ) -> List[HistoricalAccuracy]:
        """Retrieve historical accuracy metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM accuracy_metrics"
                params = []
                conditions = []

                if stock_symbol:
                    conditions.append("stock_symbol = ?")
                    params.append(stock_symbol)

                if time_period_days:
                    conditions.append("time_period_days = ?")
                    params.append(time_period_days)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY calculated_at DESC"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                metrics = []
                for row in rows:
                    accuracy = HistoricalAccuracy(
                        stock_symbol=row[1],
                        time_period_days=row[2],
                        total_predictions=row[3],
                        correct_predictions=row[4],
                        accuracy_rate=row[5],
                        average_confidence=row[6],
                        calculated_at=datetime.fromisoformat(row[7]),
                    )
                    metrics.append(accuracy)

                return metrics

        except sqlite3.Error as e:
            print(f"Error retrieving accuracy metrics: {e}")
            return []

    def calculate_historical_accuracy(
        self, stock_symbol: str, time_period_days: int = 30
    ) -> Optional[HistoricalAccuracy]:
        """Calculate historical accuracy for a stock over a time period."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_period_days)

            # Get predictions in the time period
            predictions = self.retrieve_predictions(
                stock_symbol=stock_symbol, date_range=(start_date, end_date)
            )

            if not predictions:
                return None

            # Get outcomes for these predictions
            correct_predictions = 0
            total_confidence = 0.0

            for prediction in predictions:
                outcomes = self.retrieve_outcomes(prediction_id=prediction.article_id)
                if outcomes:
                    # Use the first outcome (could be enhanced to use specific time horizon)
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
            print(f"Error calculating historical accuracy: {e}")
            return None

    def get_similar_historical_predictions(
        self, stock_symbol: str, sentiment_score: float, lookback_days: int = 90
    ) -> List[Tuple[MarketPrediction, Optional[MarketOutcome]]]:
        """Get similar historical predictions for influence calculation."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # Get predictions for the same stock in the lookback period
            predictions = self.retrieve_predictions(
                stock_symbol=stock_symbol, date_range=(start_date, end_date)
            )

            # Filter predictions with similar sentiment (within 0.2 range)
            similar_predictions = []
            for prediction in predictions:
                # For now, we'll use impact_magnitude as a proxy for sentiment similarity
                # In a real implementation, you'd store sentiment scores with predictions
                outcomes = self.retrieve_outcomes(prediction_id=prediction.article_id)
                outcome = outcomes[0] if outcomes else None
                similar_predictions.append((prediction, outcome))

            return similar_predictions

        except Exception as e:
            print(f"Error retrieving similar historical predictions: {e}")
            return []

    def cleanup_old_data(self, retention_days: int = 365) -> bool:
        """Clean up old data based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            with sqlite3.connect(self.db_path) as conn:
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

                conn.commit()
                return True

        except sqlite3.Error as e:
            print(f"Error cleaning up old data: {e}")
            return False

    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about the database contents."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                stats = {}

                # Count predictions
                cursor.execute("SELECT COUNT(*) FROM predictions")
                stats["total_predictions"] = cursor.fetchone()[0]

                # Count outcomes
                cursor.execute("SELECT COUNT(*) FROM outcomes")
                stats["total_outcomes"] = cursor.fetchone()[0]

                # Count accuracy metrics
                cursor.execute("SELECT COUNT(*) FROM accuracy_metrics")
                stats["total_accuracy_records"] = cursor.fetchone()[0]

                # Count unique stocks
                cursor.execute("SELECT COUNT(DISTINCT stock_symbol) FROM predictions")
                stats["unique_stocks"] = cursor.fetchone()[0]

                return stats

        except sqlite3.Error as e:
            print(f"Error getting database stats: {e}")
            return {}
