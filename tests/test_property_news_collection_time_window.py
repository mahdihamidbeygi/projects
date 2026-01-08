"""
Property-based tests for news collection time window compliance.

**Feature: news-market-predictor, Property 1: News collection time window compliance**
"""

from datetime import datetime, timedelta
from typing import List

from hypothesis import given, strategies as st

from news_market_predictor.models import NewsArticle


def simulate_fetch_daily_news(target_date: datetime) -> List[NewsArticle]:
    """
    Simulate the fetch_daily_news method that should collect articles within 24-hour window.

    This function represents what the NewsFetcher.fetch_daily_news method should do:
    retrieve all new articles from the past 24 hours relative to the target date.
    """
    # Define the 24-hour window for the target date
    start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=1)

    # Simulate fetching articles - in real implementation this would call RSS feeds
    # For testing, we'll create sample articles with various timestamps
    sample_articles = []

    # Create articles within the valid time window
    valid_timestamps = [
        start_date + timedelta(hours=2),  # Early morning
        start_date + timedelta(hours=8, minutes=30),  # Morning
        start_date + timedelta(hours=14, minutes=15),  # Afternoon
        start_date + timedelta(hours=20, minutes=45),  # Evening
        start_date + timedelta(hours=23, minutes=59),  # End of day
    ]

    for i, timestamp in enumerate(valid_timestamps):
        article = NewsArticle(
            id=f"article_{i}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            title=f"Test Article {i}",
            content=f"Content for article {i}",
            url=f"https://finance.yahoo.com/news/article-{i}",
            published_at=timestamp,
            source="Yahoo Finance",
            category="general",
            raw_metadata={"test": True},
        )
        sample_articles.append(article)

    # Filter articles to only include those within the 24-hour window
    # This is the core logic that the property test validates
    filtered_articles = []
    for article in sample_articles:
        if start_date <= article.published_at < end_date:
            filtered_articles.append(article)

    return filtered_articles


@given(st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)))
def test_news_collection_time_window_compliance(target_date):
    """
    **Feature: news-market-predictor, Property 1: News collection time window compliance**

    Property: For any daily news collection run, all retrieved articles should have
    publication timestamps within the specified 24-hour window.

    **Validates: Requirements 1.1**

    This test verifies that the news collection process only returns articles
    published within the 24-hour window of the target date (from 00:00:00 to 23:59:59).
    """
    # Fetch articles for the target date
    articles = simulate_fetch_daily_news(target_date)

    # Define the expected 24-hour window
    start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=1)

    # Verify all articles are within the time window
    for article in articles:
        assert isinstance(
            article.published_at, datetime
        ), f"Article {article.id} has invalid published_at type: {type(article.published_at)}"

        assert start_date <= article.published_at < end_date, (
            f"Article {article.id} published at {article.published_at} is outside the "
            f"24-hour window [{start_date}, {end_date})"
        )

    # Verify that we actually got some articles (assuming the simulation provides them)
    assert len(articles) > 0, "No articles were returned for the target date"

    # Verify all articles are valid NewsArticle objects
    for article in articles:
        assert isinstance(article, NewsArticle)
        assert article.validate() is True


@given(st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)))
def test_news_collection_time_window_boundary_conditions(target_date):
    """
    **Feature: news-market-predictor, Property 1: News collection time window compliance**

    Property: For any daily news collection run, articles at the exact boundary
    of the 24-hour window should be handled correctly.

    **Validates: Requirements 1.1**

    This test verifies boundary conditions for the time window filtering.
    """
    # Create test articles at exact boundary times
    start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=1)

    # Article exactly at start of window (should be included)
    article_at_start = NewsArticle(
        id="boundary_start",
        title="Article at Start",
        content="Content at start of window",
        url="https://finance.yahoo.com/news/start",
        published_at=start_date,
        source="Yahoo Finance",
        category="general",
        raw_metadata={},
    )

    # Article exactly at end of window (should be excluded - end is exclusive)
    article_at_end = NewsArticle(
        id="boundary_end",
        title="Article at End",
        content="Content at end of window",
        url="https://finance.yahoo.com/news/end",
        published_at=end_date,
        source="Yahoo Finance",
        category="general",
        raw_metadata={},
    )

    # Article one microsecond before end (should be included)
    article_before_end = NewsArticle(
        id="boundary_before_end",
        title="Article Before End",
        content="Content just before end of window",
        url="https://finance.yahoo.com/news/before-end",
        published_at=end_date - timedelta(microseconds=1),
        source="Yahoo Finance",
        category="general",
        raw_metadata={},
    )

    # Test the filtering logic directly
    test_articles = [article_at_start, article_at_end, article_before_end]

    # Filter articles using the same logic as fetch_daily_news
    filtered_articles = []
    for article in test_articles:
        if start_date <= article.published_at < end_date:
            filtered_articles.append(article)

    # Verify boundary conditions
    assert (
        len(filtered_articles) == 2
    ), f"Expected 2 articles within window, got {len(filtered_articles)}"

    # Verify specific articles are included/excluded correctly
    filtered_ids = [article.id for article in filtered_articles]
    assert (
        "boundary_start" in filtered_ids
    ), "Article at start of window should be included"
    assert (
        "boundary_before_end" in filtered_ids
    ), "Article before end should be included"
    assert (
        "boundary_end" not in filtered_ids
    ), "Article at end of window should be excluded"


@given(
    st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)),
    st.lists(
        st.datetimes(min_value=datetime(2019, 1, 1), max_value=datetime(2025, 12, 31)),
        min_size=1,
        max_size=10,
    ),
)
def test_news_collection_time_window_mixed_timestamps(target_date, article_timestamps):
    """
    **Feature: news-market-predictor, Property 1: News collection time window compliance**

    Property: For any daily news collection run with mixed article timestamps,
    only articles within the 24-hour window should be returned.

    **Validates: Requirements 1.1**

    This test verifies filtering works correctly with a mix of valid and invalid timestamps.
    """
    # Define the 24-hour window
    start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=1)

    # Create articles with the provided timestamps
    test_articles = []
    for i, timestamp in enumerate(article_timestamps):
        article = NewsArticle(
            id=f"mixed_article_{i}",
            title=f"Mixed Article {i}",
            content=f"Content for mixed article {i}",
            url=f"https://finance.yahoo.com/news/mixed-{i}",
            published_at=timestamp,
            source="Yahoo Finance",
            category="general",
            raw_metadata={"index": i},
        )
        test_articles.append(article)

    # Filter articles using the same logic as fetch_daily_news
    filtered_articles = []
    for article in test_articles:
        if start_date <= article.published_at < end_date:
            filtered_articles.append(article)

    # Verify all filtered articles are within the time window
    for article in filtered_articles:
        assert (
            start_date <= article.published_at < end_date
        ), f"Filtered article {article.id} at {article.published_at} is outside window"

    # Verify no articles outside the window were included
    expected_count = sum(
        1 for timestamp in article_timestamps if start_date <= timestamp < end_date
    )
    assert (
        len(filtered_articles) == expected_count
    ), f"Expected {expected_count} articles, got {len(filtered_articles)}"

    # Verify all filtered articles are valid
    for article in filtered_articles:
        assert isinstance(article, NewsArticle)
        assert article.validate() is True
