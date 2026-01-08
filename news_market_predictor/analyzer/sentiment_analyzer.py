"""
Sentiment analyzer implementation using VADER sentiment analysis.
"""

import logging
import re
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..interfaces import SentimentAnalyzer
from ..models import SentimentAnalysis
from ..exceptions import SentimentAnalysisError


logger = logging.getLogger(__name__)


class VaderSentimentAnalyzer(SentimentAnalyzer):
    """Concrete implementation of sentiment analysis using VADER."""

    def __init__(self):
        """Initialize the VADER sentiment analyzer."""
        try:
            self.analyzer = SentimentIntensityAnalyzer()
            # Market-specific keywords for tone detection
            self.bullish_keywords = {
                "growth",
                "profit",
                "revenue",
                "earnings",
                "beat",
                "exceed",
                "strong",
                "positive",
                "gain",
                "rise",
                "increase",
                "up",
                "bull",
                "buy",
                "upgrade",
                "outperform",
                "success",
                "expansion",
                "acquisition",
                "merger",
                "dividend",
            }
            self.bearish_keywords = {
                "loss",
                "decline",
                "fall",
                "drop",
                "down",
                "bear",
                "sell",
                "downgrade",
                "underperform",
                "miss",
                "below",
                "weak",
                "negative",
                "concern",
                "risk",
                "bankruptcy",
                "layoff",
                "cut",
                "reduce",
                "warning",
                "investigation",
            }
            logger.info("VADER sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VADER sentiment analyzer: {e}")
            raise SentimentAnalysisError(
                f"Sentiment analyzer initialization failed: {e}"
            ) from e

    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze sentiment of article text.

        Args:
            text: Text to analyze for sentiment

        Returns:
            SentimentAnalysis object with sentiment scores and metadata

        Raises:
            SentimentAnalysisError: If sentiment analysis fails
        """
        try:
            if not isinstance(text, str):
                raise SentimentAnalysisError(f"Expected string input, got {type(text)}")

            if not text.strip():
                logger.warning(
                    "Empty or whitespace-only text provided for sentiment analysis"
                )
                # Return neutral sentiment for empty text
                return SentimentAnalysis(
                    article_id="",  # Will be set by caller
                    sentiment_score=0.0,
                    confidence=0.0,
                    key_phrases=[],
                    market_tone="neutral",
                )

            # Get VADER scores
            scores = self.analyzer.polarity_scores(text)

            # Extract compound score as main sentiment (-1 to 1)
            sentiment_score = scores["compound"]

            # Ensure score is within bounds
            sentiment_score = max(-1.0, min(1.0, sentiment_score))

            # Calculate confidence based on the strength of sentiment
            confidence = self._calculate_confidence(text, sentiment_score)

            # Extract key phrases
            key_phrases = self._extract_key_phrases(text)

            # Detect market tone
            market_tone = self.detect_market_tone(text, sentiment_score)

            result = SentimentAnalysis(
                article_id="",  # Will be set by caller
                sentiment_score=sentiment_score,
                confidence=confidence,
                key_phrases=key_phrases,
                market_tone=market_tone,
            )

            logger.debug(
                f"Sentiment analysis completed: score={sentiment_score:.3f}, confidence={confidence:.3f}, tone={market_tone}"
            )
            return result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            # Return neutral sentiment on error
            return SentimentAnalysis(
                article_id="",
                sentiment_score=0.0,
                confidence=0.0,
                key_phrases=[],
                market_tone="neutral",
            )

    def calculate_confidence(self, text: str, sentiment_score: float) -> float:
        """
        Calculate confidence in sentiment analysis.

        Args:
            text: Original text that was analyzed
            sentiment_score: The calculated sentiment score

        Returns:
            Confidence level between 0.0 and 1.0
        """
        return self._calculate_confidence(text, sentiment_score)

    def _calculate_confidence(self, text: str, sentiment_score: float) -> float:
        """
        Internal method to calculate confidence in sentiment analysis.

        Args:
            text: Original text that was analyzed
            sentiment_score: The calculated sentiment score

        Returns:
            Confidence level between 0.0 and 1.0
        """
        try:
            if not text.strip():
                return 0.0

            # Base confidence on absolute sentiment score
            base_confidence = abs(sentiment_score)

            # Adjust based on text length (longer text generally more reliable)
            text_length_factor = min(
                1.0, len(text.split()) / 50.0
            )  # Normalize to 50 words

            # Adjust based on presence of strong sentiment words
            strong_words = len(
                [
                    word
                    for word in text.lower().split()
                    if word in self.bullish_keywords or word in self.bearish_keywords
                ]
            )
            strong_word_factor = min(
                1.0, strong_words / 10.0
            )  # Normalize to 10 strong words

            # Combine factors
            confidence = (
                base_confidence * 0.6
                + text_length_factor * 0.2
                + strong_word_factor * 0.2
            )

            # Ensure bounds
            confidence = max(0.0, min(1.0, confidence))

            return confidence

        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0

    def detect_market_tone(self, text: str, sentiment_score: float) -> str:
        """
        Detect market tone (bullish, bearish, neutral).

        Args:
            text: Text to analyze for market tone
            sentiment_score: Calculated sentiment score

        Returns:
            Market tone: 'bullish', 'bearish', or 'neutral'
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return "neutral"

            text_lower = text.lower()

            # Count market-specific keywords
            bullish_count = sum(
                1 for word in self.bullish_keywords if word in text_lower
            )
            bearish_count = sum(
                1 for word in self.bearish_keywords if word in text_lower
            )

            # Combine keyword analysis with sentiment score
            if sentiment_score > 0.1 and bullish_count > bearish_count:
                return "bullish"
            elif sentiment_score < -0.1 and bearish_count > bullish_count:
                return "bearish"
            elif bullish_count > bearish_count + 1:  # Strong bullish keyword presence
                return "bullish"
            elif bearish_count > bullish_count + 1:  # Strong bearish keyword presence
                return "bearish"
            else:
                return "neutral"

        except Exception as e:
            logger.error(f"Market tone detection failed: {e}")
            return "neutral"

    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from text that might be relevant for sentiment.

        Args:
            text: Text to extract phrases from

        Returns:
            List of key phrases
        """
        try:
            if not text.strip():
                return []

            key_phrases = []
            text_lower = text.lower()

            # Extract market-relevant phrases
            all_keywords = self.bullish_keywords.union(self.bearish_keywords)

            # Find phrases containing market keywords
            sentences = re.split(r"[.!?]+", text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Minimum sentence length
                    sentence_lower = sentence.lower()
                    if any(keyword in sentence_lower for keyword in all_keywords):
                        # Truncate long sentences
                        if len(sentence) > 100:
                            sentence = sentence[:97] + "..."
                        key_phrases.append(sentence)

            # Limit to top 5 phrases
            return key_phrases[:5]

        except Exception as e:
            logger.error(f"Key phrase extraction failed: {e}")
            return []
