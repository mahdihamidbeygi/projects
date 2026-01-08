"""
Property-based tests for sentiment score bounds validation.

**Feature: news-market-predictor, Property 6: Sentiment score bounds**
"""

from hypothesis import given, strategies as st

from news_market_predictor.analyzer.sentiment_analyzer import VaderSentimentAnalyzer
from news_market_predictor.models import SentimentAnalysis


@given(
    st.text(
        min_size=0, max_size=10000
    )  # Test with various text lengths including empty
)
def test_sentiment_score_bounds_for_any_text(text):
    """
    **Feature: news-market-predictor, Property 6: Sentiment score bounds**

    Property: For any valid news article, the generated sentiment score should always
    be between -1.0 and 1.0 inclusive.

    **Validates: Requirements 2.1**

    This test verifies that regardless of input text content, the sentiment analyzer
    always produces sentiment scores within the valid range of -1.0 to 1.0.
    """
    # Initialize the sentiment analyzer
    analyzer = VaderSentimentAnalyzer()

    # Analyze sentiment for the given text
    result = analyzer.analyze_sentiment(text)

    # Verify the result is a SentimentAnalysis object
    assert isinstance(result, SentimentAnalysis)

    # Verify sentiment score is within bounds
    assert isinstance(result.sentiment_score, (int, float))
    assert -1.0 <= result.sentiment_score <= 1.0

    # Verify the sentiment score is a valid number (not NaN or infinity)
    assert not (result.sentiment_score != result.sentiment_score)  # Check for NaN
    assert result.sentiment_score != float("inf")
    assert result.sentiment_score != float("-inf")


@given(
    st.text(min_size=1, max_size=5000).filter(
        lambda x: x.strip()
    )  # Non-empty text only
)
def test_sentiment_score_bounds_for_valid_content(text):
    """
    **Feature: news-market-predictor, Property 6: Sentiment score bounds**

    Property: For any non-empty text content, the sentiment score should be within
    bounds and the analysis should complete successfully.

    **Validates: Requirements 2.1**

    This test focuses on valid, non-empty text content to ensure proper sentiment
    analysis while maintaining score bounds.
    """
    # Initialize the sentiment analyzer
    analyzer = VaderSentimentAnalyzer()

    # Analyze sentiment for the given text
    result = analyzer.analyze_sentiment(text)

    # Verify sentiment score bounds
    assert -1.0 <= result.sentiment_score <= 1.0

    # Verify other fields are also properly bounded
    assert 0.0 <= result.confidence <= 1.0
    assert result.market_tone in ["bullish", "bearish", "neutral"]
    assert isinstance(result.key_phrases, list)

    # Set article_id for validation (analyzer leaves it empty by design)
    result.article_id = "test_article_123"

    # Verify the result passes model validation
    assert result.validate() is True


@given(
    st.one_of(
        st.text(min_size=0, max_size=0),  # Empty string
        st.text().filter(lambda x: not x.strip()),  # Whitespace only
        st.just(None),  # None value (should be handled gracefully)
    )
)
def test_sentiment_score_bounds_for_edge_cases(text):
    """
    **Feature: news-market-predictor, Property 6: Sentiment score bounds**

    Property: For edge cases like empty text, whitespace, or None values,
    the sentiment analyzer should handle them gracefully and still return
    scores within valid bounds.

    **Validates: Requirements 2.1**

    This test ensures robust handling of edge cases while maintaining score bounds.
    """
    # Initialize the sentiment analyzer
    analyzer = VaderSentimentAnalyzer()

    # Handle None input by converting to empty string
    input_text = text if text is not None else ""

    # Analyze sentiment - should not raise exceptions
    result = analyzer.analyze_sentiment(input_text)

    # Verify sentiment score bounds are maintained even for edge cases
    assert isinstance(result.sentiment_score, (int, float))
    assert -1.0 <= result.sentiment_score <= 1.0

    # For empty/whitespace input, expect neutral sentiment
    if not input_text or not input_text.strip():
        assert result.sentiment_score == 0.0
        assert result.confidence == 0.0
        assert result.market_tone == "neutral"
        assert result.key_phrases == []


@given(
    st.text(min_size=1, max_size=1000).map(
        lambda x: x
        + " "
        + " ".join(
            [
                "excellent",
                "amazing",
                "terrible",
                "awful",
                "great",
                "horrible",
                "fantastic",
                "disgusting",
                "wonderful",
                "dreadful",
                "outstanding",
                "pathetic",
            ]
        )
    )
)
def test_sentiment_score_bounds_for_extreme_sentiment(text):
    """
    **Feature: news-market-predictor, Property 6: Sentiment score bounds**

    Property: Even for text with extreme positive or negative sentiment words,
    the sentiment score should remain within the -1.0 to 1.0 bounds.

    **Validates: Requirements 2.1**

    This test verifies that strong sentiment words don't cause the analyzer
    to produce out-of-bounds scores.
    """
    # Initialize the sentiment analyzer
    analyzer = VaderSentimentAnalyzer()

    # Analyze sentiment for text with extreme sentiment words
    result = analyzer.analyze_sentiment(text)

    # Verify bounds are maintained even with extreme sentiment
    assert -1.0 <= result.sentiment_score <= 1.0

    # Verify the score is still a valid number
    assert isinstance(result.sentiment_score, (int, float))
    assert not (result.sentiment_score != result.sentiment_score)  # Not NaN

    # Set article_id for validation (analyzer leaves it empty by design)
    result.article_id = "test_article_456"

    # Verify other components are also valid
    assert result.validate() is True


@given(st.text(min_size=10, max_size=2000).filter(lambda x: x.strip()))
def test_sentiment_score_consistency_with_bounds(text):
    """
    **Feature: news-market-predictor, Property 6: Sentiment score bounds**

    Property: Multiple analyses of the same text should produce the same
    sentiment score within bounds, ensuring consistency.

    **Validates: Requirements 2.1**

    This test verifies that the sentiment analyzer produces consistent results
    while maintaining score bounds.
    """
    # Initialize the sentiment analyzer
    analyzer = VaderSentimentAnalyzer()

    # Analyze the same text multiple times
    result1 = analyzer.analyze_sentiment(text)
    result2 = analyzer.analyze_sentiment(text)
    result3 = analyzer.analyze_sentiment(text)

    # Verify all results have scores within bounds
    assert -1.0 <= result1.sentiment_score <= 1.0
    assert -1.0 <= result2.sentiment_score <= 1.0
    assert -1.0 <= result3.sentiment_score <= 1.0

    # Verify consistency - same input should produce same output
    assert result1.sentiment_score == result2.sentiment_score == result3.sentiment_score
    assert result1.confidence == result2.confidence == result3.confidence
    assert result1.market_tone == result2.market_tone == result3.market_tone

    # Set article_id for validation (analyzer leaves it empty by design)
    result1.article_id = "test_article_789"
    result2.article_id = "test_article_789"
    result3.article_id = "test_article_789"

    # Verify all results pass validation
    assert result1.validate() is True
    assert result2.validate() is True
    assert result3.validate() is True
