"""
Base interfaces and abstract classes for the News Market Predictor system.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from .models import (
    NewsArticle,
    SentimentAnalysis,
    ExtractedEntity,
    MarketPrediction,
    MarketOutcome,
    HistoricalAccuracy,
)


class NewsFetcher(ABC):
    """Abstract base class for news fetching components."""

    @abstractmethod
    def fetch_daily_news(self, date: Optional[datetime] = None) -> List[NewsArticle]:
        """Fetch news articles for a specific date (defaults to today)."""
        pass

    @abstractmethod
    def parse_article_content(
        self, raw_content: str, metadata: Dict[str, Any]
    ) -> NewsArticle:
        """Parse raw article content into structured NewsArticle."""
        pass

    @abstractmethod
    def deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title and content similarity."""
        pass


class ContentProcessor(ABC):
    """Abstract base class for content processing components."""

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Clean and normalize article text."""
        pass

    @abstractmethod
    def extract_metadata(self, article: NewsArticle) -> Dict[str, Any]:
        """Extract metadata from article content."""
        pass

    @abstractmethod
    def validate_content(self, article: NewsArticle) -> bool:
        """Validate that article content is processable."""
        pass


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analysis components."""

    @abstractmethod
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment of article text."""
        pass

    @abstractmethod
    def calculate_confidence(self, text: str, sentiment_score: float) -> float:
        """Calculate confidence in sentiment analysis."""
        pass

    @abstractmethod
    def detect_market_tone(self, text: str, sentiment_score: float) -> str:
        """Detect market tone (bullish, bearish, neutral)."""
        pass


class EntityExtractor(ABC):
    """Abstract base class for entity extraction components."""

    @abstractmethod
    def extract_stock_symbols(self, text: str) -> List[ExtractedEntity]:
        """Extract stock symbols from article text."""
        pass

    @abstractmethod
    def identify_companies(self, text: str) -> List[ExtractedEntity]:
        """Identify company names in article text."""
        pass

    @abstractmethod
    def find_financial_metrics(self, text: str) -> List[ExtractedEntity]:
        """Find financial metrics (earnings, revenue, etc.) in text."""
        pass


class MarketPredictor(ABC):
    """Abstract base class for market prediction components."""

    @abstractmethod
    def predict_impact(
        self,
        article: NewsArticle,
        sentiment: SentimentAnalysis,
        entities: List[ExtractedEntity],
    ) -> List[MarketPrediction]:
        """Generate market impact predictions."""
        pass

    @abstractmethod
    def calculate_confidence(self, prediction_data: Dict[str, Any]) -> float:
        """Calculate confidence level for predictions."""
        pass

    @abstractmethod
    def aggregate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple prediction signals."""
        pass


class ResultsAggregator(ABC):
    """Abstract base class for results aggregation components."""

    @abstractmethod
    def aggregate_predictions(
        self, predictions: List[MarketPrediction]
    ) -> List[MarketPrediction]:
        """Aggregate predictions for multiple stocks and time periods."""
        pass

    @abstractmethod
    def weight_by_confidence(
        self, predictions: List[MarketPrediction]
    ) -> List[MarketPrediction]:
        """Weight predictions by confidence levels."""
        pass

    @abstractmethod
    def calculate_accuracy_metrics(
        self, predictions: List[MarketPrediction], actual_outcomes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate accuracy metrics for historical predictions."""
        pass


class DataStorage(ABC):
    """Abstract base class for data storage components."""

    @abstractmethod
    def store_article(self, article: NewsArticle) -> bool:
        """Store a news article."""
        pass

    @abstractmethod
    def store_prediction(self, prediction: MarketPrediction) -> bool:
        """Store a market prediction."""
        pass

    @abstractmethod
    def retrieve_articles(
        self, date_range: Optional[tuple] = None
    ) -> List[NewsArticle]:
        """Retrieve articles within date range."""
        pass

    @abstractmethod
    def retrieve_predictions(
        self, stock_symbol: Optional[str] = None
    ) -> List[MarketPrediction]:
        """Retrieve predictions for a specific stock or all stocks."""
        pass


class HistoricalDataInterface(ABC):
    """Abstract base class for historical data management."""

    @abstractmethod
    def store_outcome(self, outcome: MarketOutcome) -> bool:
        """Store a market outcome."""
        pass

    @abstractmethod
    def store_accuracy_metrics(self, accuracy: HistoricalAccuracy) -> bool:
        """Store historical accuracy metrics."""
        pass

    @abstractmethod
    def calculate_historical_accuracy(
        self, stock_symbol: str, time_period_days: int = 30
    ) -> Optional[HistoricalAccuracy]:
        """Calculate historical accuracy for a stock over a time period."""
        pass

    @abstractmethod
    def get_similar_historical_predictions(
        self, stock_symbol: str, sentiment_score: float, lookback_days: int = 90
    ) -> List[Tuple[MarketPrediction, Optional[MarketOutcome]]]:
        """Get similar historical predictions for influence calculation."""
        pass

    @abstractmethod
    def cleanup_old_data(self, retention_days: int = 365) -> bool:
        """Clean up old data based on retention policy."""
        pass
