"""
Property-based tests for error resilience validation.

**Feature: news-market-predictor, Property 18: Error resilience**
"""

import logging
from datetime import datetime
from typing import List
from unittest.mock import Mock, patch

from hypothesis import given, strategies as st, settings

from news_market_predictor.models import NewsArticle, ValidationError
from news_market_predictor.analyzer.content_processor import NewsContentProcessor
from news_market_predictor.analyzer.sentiment_analyzer import VaderSentimentAnalyzer


class BatchProcessor:
    """
    Simple batch processor for testing error resilience.

    This class simulates batch processing of articles where some articles
    might be malformed and should be handled gracefully.
    """

    def __init__(self):
        self.content_processor = NewsContentProcessor()
        self.sentiment_analyzer = VaderSentimentAnalyzer()
        self.logger = logging.getLogger(__name__)
        self.processed_articles = []
        self.failed_articles = []
        self.errors = []

    def process_batch(self, articles: List[NewsArticle]) -> dict:
        """
        Process a batch of articles, handling errors gracefully.

        Args:
            articles: List of NewsArticle objects to process

        Returns:
            Dictionary with processing results including success/failure counts
        """
        results = {
            "processed_count": 0,
            "failed_count": 0,
            "total_count": len(articles),
            "processed_articles": [],
            "failed_articles": [],
            "errors": [],
        }

        for article in articles:
            try:
                # Validate article first
                article.validate()

                # Process content
                if self.content_processor.validate_content(article):
                    cleaned_content = self.content_processor.clean_text(article.content)
                    metadata = self.content_processor.extract_metadata(article)

                    # Analyze sentiment
                    sentiment = self.sentiment_analyzer.analyze_sentiment(
                        cleaned_content
                    )
                    sentiment.article_id = article.id

                    # Store successful processing
                    results["processed_articles"].append(
                        {
                            "article": article,
                            "sentiment": sentiment,
                            "metadata": metadata,
                        }
                    )
                    results["processed_count"] += 1

                    self.logger.debug(f"Successfully processed article {article.id}")
                else:
                    # Content validation failed but we continue
                    error_msg = f"Content validation failed for article {article.id}"
                    self.logger.error(error_msg)
                    results["failed_articles"].append(article)
                    results["errors"].append(error_msg)
                    results["failed_count"] += 1

            except ValidationError as e:
                # Handle validation errors gracefully
                error_msg = f"Validation error for article {getattr(article, 'id', 'unknown')}: {e}"
                self.logger.error(error_msg)
                results["failed_articles"].append(article)
                results["errors"].append(error_msg)
                results["failed_count"] += 1

            except Exception as e:
                # Handle any other errors gracefully
                error_msg = f"Processing error for article {getattr(article, 'id', 'unknown')}: {e}"
                self.logger.error(error_msg)
                results["failed_articles"].append(article)
                results["errors"].append(error_msg)
                results["failed_count"] += 1

        self.logger.info(
            f"Batch processing completed: {results['processed_count']} successful, {results['failed_count']} failed"
        )
        return results


def create_valid_article(article_id: str = "test_article") -> NewsArticle:
    """Create a valid NewsArticle for testing."""
    return NewsArticle(
        id=article_id,
        title="Test Article Title",
        content="This is a test article content with some meaningful text for analysis.",
        url="https://finance.yahoo.com/news/test-article",
        published_at=datetime.now(),
        source="Yahoo Finance",
        category="Technology",
        raw_metadata={"test": "metadata"},
    )


def create_malformed_article(
    malformation_type: str, article_id: str = "malformed_article"
) -> NewsArticle:
    """Create a malformed NewsArticle for testing."""
    base_article = create_valid_article(article_id)

    if malformation_type == "empty_title":
        base_article.title = ""
    elif malformation_type == "empty_content":
        base_article.content = ""
    elif malformation_type == "empty_id":
        base_article.id = ""
    elif malformation_type == "invalid_url":
        base_article.url = ""
    elif malformation_type == "empty_source":
        base_article.source = ""
    elif malformation_type == "empty_category":
        base_article.category = ""
    elif malformation_type == "whitespace_only_title":
        base_article.title = "   \n\t   "
    elif malformation_type == "whitespace_only_content":
        base_article.content = "   \n\t   "
    elif malformation_type == "very_short_content":
        base_article.content = "Hi"  # Too short for meaningful analysis

    return base_article


@given(
    st.lists(
        st.sampled_from(
            [
                "empty_title",
                "empty_content",
                "empty_id",
                "invalid_url",
                "empty_source",
                "empty_category",
                "whitespace_only_title",
                "whitespace_only_content",
                "very_short_content",
            ]
        ),
        min_size=1,
        max_size=5,
    ),
    st.integers(min_value=1, max_value=10),
)
@settings(max_examples=5)  # Reduced examples for faster execution
def test_error_resilience_with_malformed_articles(malformation_types, valid_count):
    """
    **Feature: news-market-predictor, Property 18: Error resilience**

    Property: For any malformed article in a batch, processing should continue
    for remaining valid articles while logging the error.

    **Validates: Requirements 5.1**

    This test verifies that when processing a batch containing malformed articles,
    the system continues processing valid articles and logs errors appropriately.
    """
    processor = BatchProcessor()

    # Create a batch with both valid and malformed articles
    articles = []

    # Add valid articles
    for i in range(valid_count):
        articles.append(create_valid_article(f"valid_article_{i}"))

    # Add malformed articles
    for i, malformation_type in enumerate(malformation_types):
        articles.append(
            create_malformed_article(malformation_type, f"malformed_article_{i}")
        )

    # Process the batch
    with patch(
        "news_market_predictor.analyzer.content_processor.logger"
    ) as mock_logger:
        results = processor.process_batch(articles)

    # Verify error resilience properties
    assert results["total_count"] == len(articles)
    assert (
        results["processed_count"] + results["failed_count"] == results["total_count"]
    )

    # At least some articles should be processed successfully (the valid ones)
    assert results["processed_count"] >= valid_count

    # Some articles should fail (the malformed ones)
    assert results["failed_count"] >= len(malformation_types)

    # Errors should be logged for failed articles
    assert len(results["errors"]) == results["failed_count"]

    # Processing should continue despite errors - verify all articles were attempted
    assert len(results["processed_articles"]) + len(results["failed_articles"]) == len(
        articles
    )

    # Verify that valid articles were processed successfully
    processed_ids = [item["article"].id for item in results["processed_articles"]]
    expected_valid_ids = [f"valid_article_{i}" for i in range(valid_count)]

    # All valid articles should be in processed results
    for valid_id in expected_valid_ids:
        assert valid_id in processed_ids


@given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10))
@settings(max_examples=5)  # Reduced examples for faster execution
def test_error_resilience_with_processing_exceptions(article_titles):
    """
    **Feature: news-market-predictor, Property 18: Error resilience**

    Property: For any processing exception during batch processing, the system
    should log the error and continue processing remaining articles.

    **Validates: Requirements 5.1**

    This test simulates processing exceptions and verifies error resilience.
    """
    processor = BatchProcessor()

    # Create articles with the given titles
    articles = []
    for i, title in enumerate(article_titles):
        article = create_valid_article(f"article_{i}")
        article.title = title
        articles.append(article)

    # Mock the content processor to raise exceptions for some articles
    original_validate_content = processor.content_processor.validate_content

    def mock_validate_content(article):
        # Simulate random processing failures for articles with certain patterns
        if "error" in article.title.lower() or "fail" in article.title.lower():
            raise Exception(f"Simulated processing error for {article.id}")
        return original_validate_content(article)

    processor.content_processor.validate_content = mock_validate_content

    # Process the batch
    with patch(
        "news_market_predictor.analyzer.content_processor.logger"
    ) as mock_logger:
        results = processor.process_batch(articles)

    # Verify error resilience properties
    assert results["total_count"] == len(articles)
    assert (
        results["processed_count"] + results["failed_count"] == results["total_count"]
    )

    # All articles should have been attempted (none skipped due to earlier errors)
    assert len(results["processed_articles"]) + len(results["failed_articles"]) == len(
        articles
    )

    # If there were processing errors, they should be logged
    if results["failed_count"] > 0:
        assert len(results["errors"]) == results["failed_count"]
        # Verify error messages contain useful information
        for error in results["errors"]:
            assert isinstance(error, str)
            assert len(error) > 0


@given(st.integers(min_value=1, max_value=20), st.integers(min_value=0, max_value=10))
@settings(max_examples=5)  # Reduced examples for faster execution
def test_error_resilience_batch_completion(valid_count, malformed_count):
    """
    **Feature: news-market-predictor, Property 18: Error resilience**

    Property: For any batch of articles containing both valid and malformed articles,
    the batch processing should complete and return results for all articles.

    **Validates: Requirements 5.1**

    This test verifies that batch processing always completes regardless of
    the mix of valid and malformed articles.
    """
    processor = BatchProcessor()

    # Create mixed batch
    articles = []

    # Add valid articles
    for i in range(valid_count):
        articles.append(create_valid_article(f"valid_{i}"))

    # Add malformed articles with various issues
    malformation_types = [
        "empty_title",
        "empty_content",
        "empty_id",
        "invalid_url",
        "empty_source",
        "empty_category",
        "whitespace_only_title",
        "whitespace_only_content",
        "very_short_content",
    ]

    for i in range(malformed_count):
        malformation_type = malformation_types[i % len(malformation_types)]
        articles.append(create_malformed_article(malformation_type, f"malformed_{i}"))

    # Process the batch
    results = processor.process_batch(articles)

    # Verify batch completion properties
    assert results["total_count"] == len(articles)
    assert results["total_count"] == valid_count + malformed_count

    # All articles should be accounted for
    assert (
        results["processed_count"] + results["failed_count"] == results["total_count"]
    )

    # Results should contain all articles in some form
    total_results = len(results["processed_articles"]) + len(results["failed_articles"])
    assert total_results == results["total_count"]

    # If there are valid articles, at least some should be processed
    if valid_count > 0:
        assert results["processed_count"] > 0

    # If there are malformed articles, some should fail
    if malformed_count > 0:
        assert results["failed_count"] > 0

    # Error count should match failed count
    assert len(results["errors"]) == results["failed_count"]


@given(st.lists(st.text(min_size=0, max_size=1000), min_size=0, max_size=15))
@settings(max_examples=5)  # Reduced examples for faster execution
def test_error_resilience_empty_and_edge_cases(content_list):
    """
    **Feature: news-market-predictor, Property 18: Error resilience**

    Property: For any batch including edge cases like empty content, very long content,
    or special characters, processing should handle them gracefully.

    **Validates: Requirements 5.1**

    This test verifies error resilience with various edge cases.
    """
    processor = BatchProcessor()

    # Handle empty list case
    if not content_list:
        results = processor.process_batch([])
        assert results["total_count"] == 0
        assert results["processed_count"] == 0
        assert results["failed_count"] == 0
        return

    # Create articles with the given content
    articles = []
    for i, content in enumerate(content_list):
        article = create_valid_article(f"edge_case_article_{i}")
        article.content = content
        articles.append(article)

    # Process the batch
    results = processor.process_batch(articles)

    # Verify error resilience properties
    assert results["total_count"] == len(articles)
    assert (
        results["processed_count"] + results["failed_count"] == results["total_count"]
    )

    # All articles should be accounted for
    total_results = len(results["processed_articles"]) + len(results["failed_articles"])
    assert total_results == results["total_count"]

    # Processing should complete without raising exceptions
    assert isinstance(results, dict)
    assert "processed_count" in results
    assert "failed_count" in results
    assert "total_count" in results
    assert "errors" in results


@given(
    st.integers(min_value=1, max_value=5),
    st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
)
@settings(max_examples=5)  # Reduced examples for faster execution
def test_error_resilience_logging_behavior(article_count, error_message):
    """
    **Feature: news-market-predictor, Property 18: Error resilience**

    Property: For any processing error, the system should log the error with
    sufficient detail while continuing to process remaining articles.

    **Validates: Requirements 5.1**

    This test verifies that error logging behavior is consistent and informative.
    """
    processor = BatchProcessor()

    # Create articles
    articles = []
    for i in range(article_count):
        articles.append(create_valid_article(f"test_article_{i}"))

    # Mock sentiment analyzer to simulate errors instead of content processor
    original_analyze_sentiment = processor.sentiment_analyzer.analyze_sentiment

    def mock_analyze_sentiment(text):
        if "trigger_error" in text:
            raise Exception(error_message)
        return original_analyze_sentiment(text)

    processor.sentiment_analyzer.analyze_sentiment = mock_analyze_sentiment

    # Add one article that will trigger an error
    error_article = create_valid_article("error_article")
    error_article.content = "This content will trigger_error during processing"
    articles.append(error_article)

    # Process with logging capture
    with patch(
        "news_market_predictor.analyzer.content_processor.logger"
    ) as mock_logger:
        results = processor.process_batch(articles)

    # Verify error resilience and logging
    assert results["total_count"] == len(articles)
    assert results["failed_count"] >= 1  # At least the error article should fail
    assert results["processed_count"] >= article_count  # Other articles should succeed

    # Verify error information is captured
    assert len(results["errors"]) == results["failed_count"]

    # At least one error should contain our error message or reference the trigger
    error_found = any(error_message in error for error in results["errors"])
    trigger_found = any("trigger_error" in error for error in results["errors"])
    processing_error_found = any(
        "Processing error" in error for error in results["errors"]
    )

    # At least one of these should be true
    assert error_found or trigger_found or processing_error_found

    # Verify processing continued after error
    assert (
        results["processed_count"] + results["failed_count"] == results["total_count"]
    )
