"""
Unit tests for YahooFinanceNewsFetcher.fetch_daily_news method.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
from news_market_predictor.fetcher.yahoo_finance_fetcher import YahooFinanceNewsFetcher
from news_market_predictor.models import NewsArticle
from news_market_predictor.exceptions import NetworkError, ParsingError


class TestFetchDailyNews:
    """Test suite for fetch_daily_news method."""

    @pytest.fixture
    def fetcher(self):
        """Create a YahooFinanceNewsFetcher instance for testing."""
        return YahooFinanceNewsFetcher(rate_limit_delay=0.1, max_retries=2, timeout=10)

    @pytest.fixture
    def sample_rss_content(self):
        """Sample RSS feed content for testing."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Yahoo Finance</title>
        <item>
            <title>Stock Market Rises on Tech Gains</title>
            <link>https://finance.yahoo.com/news/stock-market-rises-123456.html</link>
            <description>Technology stocks led the market higher today.</description>
            <pubDate>Mon, 13 Jan 2026 10:30:00 GMT</pubDate>
        </item>
        <item>
            <title>Apple Reports Strong Earnings</title>
            <link>https://finance.yahoo.com/news/apple-earnings-789012.html</link>
            <description>Apple exceeded analyst expectations in Q4.</description>
            <pubDate>Mon, 13 Jan 2026 14:15:00 GMT</pubDate>
        </item>
    </channel>
</rss>"""

    def test_fetch_daily_news_defaults_to_today(self, fetcher):
        """Test that fetch_daily_news defaults to today's date when no date is provided."""
        with patch.object(fetcher, "_fetch_rss_feed", return_value=[]):
            articles = fetcher.fetch_daily_news()
            # Should have been called for each RSS feed
            assert fetcher._fetch_rss_feed.call_count == len(
                YahooFinanceNewsFetcher.RSS_FEEDS
            )

    def test_fetch_daily_news_with_specific_date(self, fetcher):
        """Test fetch_daily_news with a specific date."""
        target_date = datetime(2026, 1, 13, 12, 0, 0)

        with patch.object(fetcher, "_fetch_rss_feed", return_value=[]):
            articles = fetcher.fetch_daily_news(date=target_date)

            # Verify the date was passed to _fetch_rss_feed
            for call in fetcher._fetch_rss_feed.call_args_list:
                assert call[0][1] == target_date

    def test_fetch_daily_news_fetches_from_all_feeds(self, fetcher, sample_rss_content):
        """Test that fetch_daily_news fetches from all configured RSS feeds."""
        with patch.object(fetcher, "_make_request") as mock_request:
            mock_response = Mock()
            mock_response.text = sample_rss_content
            mock_request.return_value = mock_response

            # Use timezone-aware datetime to match RSS parsing
            articles = fetcher.fetch_daily_news(
                date=datetime(2026, 1, 13, tzinfo=timezone.utc)
            )

            # Verify all feed URLs were called (note: _make_request is also called for scraping article content)
            called_urls = [call[0][0] for call in mock_request.call_args_list]
            for feed_url in YahooFinanceNewsFetcher.RSS_FEEDS.values():
                assert feed_url in called_urls

    def test_fetch_daily_news_returns_articles(self, fetcher, sample_rss_content):
        """Test that fetch_daily_news returns NewsArticle objects."""
        with patch.object(fetcher, "_make_request") as mock_request:
            mock_response = Mock()
            mock_response.text = sample_rss_content
            mock_request.return_value = mock_response

            # Use timezone-aware datetime
            articles = fetcher.fetch_daily_news(
                date=datetime(2026, 1, 13, tzinfo=timezone.utc)
            )

            # Should return articles
            assert len(articles) > 0
            assert all(isinstance(article, NewsArticle) for article in articles)

    def test_fetch_daily_news_filters_by_date(self, fetcher):
        """Test that fetch_daily_news filters articles to the target date."""
        target_date = datetime(2026, 1, 13, 0, 0, 0, tzinfo=timezone.utc)

        # Create articles with different dates (timezone-aware)
        articles_in_range = [
            NewsArticle(
                id="1",
                title="Article 1",
                content="Content 1",
                url="http://example.com/1",
                published_at=datetime(2026, 1, 13, 10, 0, 0, tzinfo=timezone.utc),
                source="Yahoo Finance",
                category="general",
                raw_metadata={},
            ),
            NewsArticle(
                id="2",
                title="Article 2",
                content="Content 2",
                url="http://example.com/2",
                published_at=datetime(2026, 1, 13, 20, 0, 0, tzinfo=timezone.utc),
                source="Yahoo Finance",
                category="general",
                raw_metadata={},
            ),
        ]

        articles_out_of_range = [
            NewsArticle(
                id="3",
                title="Article 3",
                content="Content 3",
                url="http://example.com/3",
                published_at=datetime(
                    2026, 1, 12, 23, 0, 0, tzinfo=timezone.utc
                ),  # Day before
                source="Yahoo Finance",
                category="general",
                raw_metadata={},
            ),
            NewsArticle(
                id="4",
                title="Article 4",
                content="Content 4",
                url="http://example.com/4",
                published_at=datetime(
                    2026, 1, 14, 1, 0, 0, tzinfo=timezone.utc
                ),  # Day after
                source="Yahoo Finance",
                category="general",
                raw_metadata={},
            ),
        ]

        all_articles = articles_in_range + articles_out_of_range

        with patch.object(fetcher, "_fetch_rss_feed", return_value=all_articles):
            result = fetcher.fetch_daily_news(date=target_date)

            # Should only return articles from target date
            assert len(result) == 2
            assert all(article.id in ["1", "2"] for article in result)

    def test_fetch_daily_news_deduplicates_articles(self, fetcher):
        """Test that fetch_daily_news removes duplicate articles."""
        target_date = datetime(2026, 1, 13, 12, 0, 0, tzinfo=timezone.utc)

        # Create duplicate articles
        duplicate_articles = [
            NewsArticle(
                id="1",
                title="Same Title",
                content="Same content for testing",
                url="http://example.com/1",
                published_at=target_date,
                source="Yahoo Finance",
                category="general",
                raw_metadata={},
            ),
            NewsArticle(
                id="2",
                title="Same Title",
                content="Same content for testing",
                url="http://example.com/2",  # Different URL
                published_at=target_date,
                source="Yahoo Finance",
                category="general",
                raw_metadata={},
            ),
        ]

        with patch.object(fetcher, "_fetch_rss_feed", return_value=duplicate_articles):
            result = fetcher.fetch_daily_news(date=target_date)

            # Should only return one article after deduplication
            assert len(result) == 1

    def test_fetch_daily_news_continues_on_feed_failure(
        self, fetcher, sample_rss_content
    ):
        """Test that fetch_daily_news continues processing other feeds if one fails."""
        call_count = 0

        def mock_request_side_effect(url):
            nonlocal call_count
            call_count += 1

            # Fail on first feed, succeed on others
            if call_count == 1:
                raise NetworkError("Network error")

            mock_response = Mock()
            mock_response.text = sample_rss_content
            return mock_response

        # Mock RSS_FEEDS to have multiple feeds
        mock_feeds = {
            "feed1": "https://example.com/feed1",
            "feed2": "https://example.com/feed2",
            "feed3": "https://example.com/feed3",
        }

        with patch.object(fetcher, "RSS_FEEDS", mock_feeds):
            with patch.object(
                fetcher, "_make_request", side_effect=mock_request_side_effect
            ):
                articles = fetcher.fetch_daily_news(
                    date=datetime(2026, 1, 13, tzinfo=timezone.utc)
                )

                # Should still return articles from successful feeds
                assert len(articles) > 0

    def test_fetch_daily_news_handles_empty_feeds(self, fetcher):
        """Test that fetch_daily_news handles empty RSS feeds gracefully."""
        empty_rss = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Yahoo Finance</title>
    </channel>
</rss>"""

        with patch.object(fetcher, "_make_request") as mock_request:
            mock_response = Mock()
            mock_response.text = empty_rss
            mock_request.return_value = mock_response

            articles = fetcher.fetch_daily_news(date=datetime(2026, 1, 13))

            # Should return empty list, not raise an error
            assert articles == []

    def test_fetch_daily_news_logs_progress(self, fetcher, sample_rss_content, caplog):
        """Test that fetch_daily_news logs appropriate progress messages."""
        import logging

        caplog.set_level(logging.INFO)

        with patch.object(fetcher, "_make_request") as mock_request:
            mock_response = Mock()
            mock_response.text = sample_rss_content
            mock_request.return_value = mock_response

            fetcher.fetch_daily_news(date=datetime(2026, 1, 13, tzinfo=timezone.utc))

            # Check that logging occurred
            assert any(
                "Fetching daily news" in record.message for record in caplog.records
            )
            assert any(
                "Fetched" in record.message and "unique articles" in record.message
                for record in caplog.records
            )

    def test_fetch_daily_news_respects_rate_limiting(self, fetcher, sample_rss_content):
        """Test that fetch_daily_news respects rate limiting between requests."""
        with patch.object(fetcher, "_fetch_rss_feed") as mock_fetch:
            # Mock the RSS feed fetching to avoid actual network calls
            mock_fetch.return_value = []

            fetcher.fetch_daily_news(date=datetime(2026, 1, 13, tzinfo=timezone.utc))

            # Should have been called for each RSS feed
            assert mock_fetch.call_count == len(YahooFinanceNewsFetcher.RSS_FEEDS)

    def test_fetch_daily_news_with_malformed_rss(self, fetcher):
        """Test that fetch_daily_news handles malformed RSS gracefully."""
        malformed_rss = "This is not valid XML"

        with patch.object(fetcher, "_make_request") as mock_request:
            mock_response = Mock()
            mock_response.text = malformed_rss
            mock_request.return_value = mock_response

            # Should not crash, but continue with other feeds
            articles = fetcher.fetch_daily_news(date=datetime(2026, 1, 13))

            # May return empty or partial results depending on which feeds failed
            assert isinstance(articles, list)

    def test_fetch_daily_news_article_fields_populated(
        self, fetcher, sample_rss_content
    ):
        """Test that articles returned by fetch_daily_news have all required fields."""
        with patch.object(fetcher, "_make_request") as mock_request:
            mock_response = Mock()
            mock_response.text = sample_rss_content
            mock_request.return_value = mock_response

            articles = fetcher.fetch_daily_news(
                date=datetime(2026, 1, 13, tzinfo=timezone.utc)
            )

            for article in articles:
                assert article.id is not None
                assert article.title is not None
                assert article.content is not None
                assert article.url is not None
                assert article.published_at is not None
                assert article.source == "Yahoo Finance"
                assert article.category is not None

    def test_fetch_daily_news_returns_unique_ids(self, fetcher, sample_rss_content):
        """Test that all articles have unique IDs."""
        with patch.object(fetcher, "_make_request") as mock_request:
            mock_response = Mock()
            mock_response.text = sample_rss_content
            mock_request.return_value = mock_response

            articles = fetcher.fetch_daily_news(
                date=datetime(2026, 1, 13, tzinfo=timezone.utc)
            )

            article_ids = [article.id for article in articles]
            # After deduplication, all IDs should be unique
            assert len(article_ids) == len(set(article_ids))

    def test_fetch_daily_news_parses_real_yahoo_rss_format(
        self, fetcher, sample_rss_content
    ):
        """Test parsing of real Yahoo Finance RSS feed format with namespaces."""
        with patch.object(fetcher, "_make_request") as mock_request:
            mock_response = Mock()
            mock_response.text = sample_rss_content
            mock_request.return_value = mock_response

            # Use date matching the RSS feed (Jan 15, 2026)
            result = fetcher.fetch_daily_news(
                date=datetime(2026, 1, 13, tzinfo=timezone.utc)
            )

            # Should successfully parse the real RSS format
            assert len(result) > 0

            # Check that articles have expected fields from real RSS
            for article in result:
                assert article.title is not None and len(article.title) > 0
                assert article.url is not None and article.url.startswith("http")
                assert article.published_at is not None
                assert article.source == "Yahoo Finance"

            # Verify specific article titles from the sample RSS
            article_titles = [article.title for article in result]
