"""
Yahoo Finance news fetcher implementation.
"""

import time
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Set
from urllib.parse import urljoin, urlparse
import re

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..interfaces import NewsFetcher
from ..models import NewsArticle
from ..exceptions import NetworkError, ParsingError, RateLimitError
from ..error_handling import (
    RateLimitHandler,
    RateLimitConfig,
    RetryConfig,
    with_retry,
    ErrorHandlingManager,
)


logger = logging.getLogger(__name__)


class YahooFinanceNewsFetcher(NewsFetcher):
    """Yahoo Finance news fetcher with RSS and web scraping capabilities."""

    # Yahoo Finance RSS feeds
    RSS_FEEDS = {
        "general": "https://finance.yahoo.com/news/rssindex",
        "markets": "https://finance.yahoo.com/topic/stock-market-news/",
        "earnings": "https://finance.yahoo.com/topic/earnings/",
        "tech": "https://finance.yahoo.com/topic/technology/",
    }

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
        user_agent: str = None,
    ):
        """
        Initialize Yahoo Finance news fetcher.

        Args:
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        # Track seen articles for deduplication
        self._seen_articles: Set[str] = set()
        self._last_request_time = 0.0

        # Configure session with retry strategy
        self.session = self._create_session()

        # Setup comprehensive error handling
        rate_limit_config = RateLimitConfig(
            requests_per_second=1.0 / rate_limit_delay,
            burst_size=3,
            cooldown_period=60.0,
        )
        retry_config = RetryConfig(
            max_attempts=max_retries
            + 1,  # +1 for initial attempt to match expected behavior
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=False,  # Disable jitter for predictable timing in tests
        )

        self.error_manager = ErrorHandlingManager(
            retry_config=retry_config, rate_limit_config=rate_limit_config
        )
        self.rate_limit_handler = self.error_manager.rate_limit_handler

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy and headers."""
        session = requests.Session()

        # Disable automatic retries - we handle retries manually
        adapter = HTTPAdapter(max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set headers
        session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        return session

    def _respect_rate_limit(self):
        """Implement rate limiting between requests."""
        self.rate_limit_handler.wait_if_needed()

    def _make_request(self, url: str) -> requests.Response:
        """
        Make HTTP request with rate limiting and error handling.

        Args:
            url: URL to fetch

        Returns:
            Response object

        Raises:
            NetworkError: If request fails after retries
            RateLimitError: If rate limited by server
        """
        # Use the error manager's retry decorator
        retry_decorator = self.error_manager.get_retry_decorator(
            exceptions=(
                NetworkError,
                RateLimitError,
                requests.exceptions.RequestException,
            )
        )

        @retry_decorator
        def make_request_with_retry():
            try:
                logger.debug("Fetching URL: %s", url)

                response = self.session.get(url, timeout=self.timeout)

                # Check for rate limiting
                if response.status_code == 429:
                    retry_after_header = response.headers.get("Retry-After")
                    retry_after = None
                    if retry_after_header:
                        try:
                            retry_after = int(retry_after_header)
                        except (ValueError, TypeError):
                            logger.warning(
                                "Invalid Retry-After header: %s", retry_after_header
                            )
                            retry_after = None

                    # Create RateLimitError with retry_after attribute for the retry decorator
                    error = RateLimitError("Rate limited by server")
                    error.retry_after = retry_after
                    raise error

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                logger.error("Request failed for %s: %s", url, e)
                raise NetworkError(f"Failed to fetch {url}: {e}") from e

        return make_request_with_retry()

    def fetch_daily_news(self, date: Optional[datetime] = None) -> List[NewsArticle]:
        """
        Fetch news articles for a specific date (defaults to today).

        Args:
            date: Target date for news articles (defaults to today)

        Returns:
            List of NewsArticle objects
        """
        if date is None:
            date = datetime.now()

        logger.info(f"Fetching daily news for {date.strftime('%Y-%m-%d')}")

        all_articles = []

        # Fetch from all RSS feeds
        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                logger.debug(f"Fetching from {feed_name} feed: {feed_url}")
                articles = self._fetch_rss_feed(feed_url, date)
                all_articles.extend(articles)
                logger.info(f"Fetched {len(articles)} articles from {feed_name} feed")
            except Exception as e:
                logger.error(f"Failed to fetch from {feed_name} feed: {e}")
                # Continue with other feeds even if one fails
                continue

        # Deduplicate articles
        unique_articles = self.deduplicate_articles(all_articles)

        # Filter articles by date (within 24 hours of target date)
        filtered_articles = self._filter_by_date(unique_articles, date)

        logger.info(
            f"Fetched {len(filtered_articles)} unique articles for {date.strftime('%Y-%m-%d')}"
        )
        return filtered_articles

    def _fetch_rss_feed(
        self, feed_url: str, target_date: datetime
    ) -> List[NewsArticle]:
        """
        Fetch articles from RSS feed.

        Args:
            feed_url: RSS feed URL
            target_date: Target date for filtering articles

        Returns:
            List of NewsArticle objects
        """
        try:
            response = self._make_request(feed_url)
            return self._parse_rss_feed(response.text, target_date)
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            raise

    def _parse_rss_feed(
        self, rss_content: str, target_date: datetime
    ) -> List[NewsArticle]:
        """
        Parse RSS feed content into NewsArticle objects.

        Args:
            rss_content: Raw RSS XML content
            target_date: Target date for filtering

        Returns:
            List of NewsArticle objects
        """
        articles = []

        try:
            root = ET.fromstring(rss_content)

            # Find all item elements
            for item in root.findall(".//item"):
                try:
                    article = self._parse_rss_item(item)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse RSS item: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"Failed to parse RSS XML: {e}")
            raise ParsingError(f"Invalid RSS XML format: {e}") from e

        return articles

    def _parse_rss_item(self, item: ET.Element) -> Optional[NewsArticle]:
        """
        Parse individual RSS item into NewsArticle.

        Args:
            item: RSS item XML element

        Returns:
            NewsArticle object or None if parsing fails
        """
        try:
            # Extract basic fields
            title_elem = item.find("title")
            title = title_elem.text.strip() if title_elem is not None else ""

            link_elem = item.find("link")
            url = link_elem.text.strip() if link_elem is not None else ""

            description_elem = item.find("description")
            description = (
                description_elem.text.strip() if description_elem is not None else ""
            )

            pub_date_elem = item.find("pubDate")
            pub_date_str = (
                pub_date_elem.text.strip() if pub_date_elem is not None else ""
            )

            # Skip if missing essential fields
            if not title or not url:
                logger.warning("Skipping RSS item with missing title or URL")
                return None

            # Parse publication date
            published_at = self._parse_rss_date(pub_date_str)
            if not published_at:
                logger.warning(f"Skipping RSS item with invalid date: {pub_date_str}")
                return None

            # Generate article ID from URL
            article_id = hashlib.md5(url.encode()).hexdigest()

            # Extract full content by scraping the article page
            full_content = self._scrape_article_content(url)
            content = full_content if full_content else description

            # Determine category from RSS feed context
            category = self._determine_category(url, title, content)

            # Build raw metadata
            raw_metadata = {
                "rss_description": description,
                "pub_date_raw": pub_date_str,
                "scraped_content": bool(full_content),
            }

            # Add any additional RSS fields
            for child in item:
                if child.tag not in ["title", "link", "description", "pubDate"]:
                    raw_metadata[f"rss_{child.tag}"] = child.text

            return NewsArticle(
                id=article_id,
                title=title,
                content=content,
                url=url,
                published_at=published_at,
                source="Yahoo Finance",
                category=category,
                raw_metadata=raw_metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse RSS item: {e}")
            return None

    def _parse_rss_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse RSS date string into datetime object.

        Args:
            date_str: RSS date string (RFC 2822 format)

        Returns:
            datetime object or None if parsing fails
        """
        if not date_str:
            return None

        try:
            # Try parsing RFC 2822 format (common in RSS)
            from email.utils import parsedate_to_datetime

            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass

        # Try other common formats
        date_formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def _scrape_article_content(self, url: str) -> Optional[str]:
        """
        Scrape full article content from URL.

        Args:
            url: Article URL

        Returns:
            Full article content or None if scraping fails
        """
        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Yahoo Finance article content selectors
            content_selectors = [
                "div[data-module='ArticleBody'] div.caas-body",
                "div.caas-body",
                "div.canvas-body",
                "div.article-body",
                "div.story-body",
            ]

            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    # Extract text and clean it
                    text = content_div.get_text(separator=" ", strip=True)
                    if text and len(text) > 100:  # Ensure we got substantial content
                        return text

            # Fallback: try to find any substantial text content
            paragraphs = soup.find_all("p")
            if paragraphs:
                text = " ".join(p.get_text(strip=True) for p in paragraphs)
                if len(text) > 100:
                    return text

            logger.warning(f"Could not extract content from {url}")
            return None

        except Exception as e:
            logger.warning(f"Failed to scrape content from {url}: {e}")
            return None

    def _determine_category(self, url: str, title: str, content: str) -> str:
        """
        Determine article category based on URL, title, and content.

        Args:
            url: Article URL
            title: Article title
            content: Article content

        Returns:
            Category string
        """
        # Check URL for category hints
        url_lower = url.lower()
        if "markets" in url_lower:
            return "markets"
        elif "earnings" in url_lower:
            return "earnings"
        elif "tech" in url_lower or "technology" in url_lower:
            return "technology"
        elif "crypto" in url_lower:
            return "cryptocurrency"

        # Check title and content for keywords
        text = f"{title} {content}".lower()

        if any(word in text for word in ["earnings", "revenue", "profit", "quarterly"]):
            return "earnings"
        elif any(word in text for word in ["market", "trading", "stock", "index"]):
            return "markets"
        elif any(word in text for word in ["tech", "technology", "software", "ai"]):
            return "technology"
        elif any(word in text for word in ["crypto", "bitcoin", "ethereum"]):
            return "cryptocurrency"
        else:
            return "general"

    def _filter_by_date(
        self, articles: List[NewsArticle], target_date: datetime
    ) -> List[NewsArticle]:
        """
        Filter articles to those within 24 hours of target date.

        Args:
            articles: List of articles to filter
            target_date: Target date

        Returns:
            Filtered list of articles
        """
        # Ensure target_date is timezone-aware for comparison
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=timezone.utc)

        start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)

        filtered = []
        for article in articles:
            # Ensure article.published_at is timezone-aware for comparison
            article_date = article.published_at
            if article_date.tzinfo is None:
                article_date = article_date.replace(tzinfo=timezone.utc)

            if start_date <= article_date < end_date:
                filtered.append(article)

        logger.debug(
            f"Filtered {len(articles)} articles to {len(filtered)} within date range"
        )

        return filtered

    def parse_article_content(
        self, raw_content: str, metadata: Dict[str, Any]
    ) -> NewsArticle:
        """
        Parse raw article content into structured NewsArticle.

        Args:
            raw_content: Raw article HTML or text content
            metadata: Additional metadata for the article

        Returns:
            NewsArticle object
        """
        try:
            soup = BeautifulSoup(raw_content, "html.parser")

            # Extract title
            title_elem = soup.find("title") or soup.find("h1")
            title = (
                title_elem.get_text(strip=True)
                if title_elem
                else metadata.get("title", "")
            )

            # Extract content
            content = self._extract_content_from_html(soup)

            # Generate ID
            url = metadata.get("url", "")
            article_id = hashlib.md5(f"{title}{url}".encode()).hexdigest()

            # Parse publication date
            published_at = metadata.get("published_at")
            if isinstance(published_at, str):
                published_at = self._parse_rss_date(published_at)
            if not published_at:
                published_at = datetime.now()

            return NewsArticle(
                id=article_id,
                title=title,
                content=content,
                url=url,
                published_at=published_at,
                source=metadata.get("source", "Yahoo Finance"),
                category=metadata.get("category", "general"),
                raw_metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse article content: {e}")
            raise ParsingError(f"Failed to parse article content: {e}") from e

    def _extract_content_from_html(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML soup."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try specific content selectors first
        content_selectors = [
            "div[data-module='ArticleBody']",
            "div.caas-body",
            "div.canvas-body",
            "div.article-body",
            "div.story-body",
            "article",
        ]

        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text(separator=" ", strip=True)
                if text and len(text) > 50:
                    return text

        # Fallback to all paragraphs
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = " ".join(p.get_text(strip=True) for p in paragraphs)
            if text:
                return text

        # Last resort: get all text
        return soup.get_text(separator=" ", strip=True)

    def deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Remove duplicate articles based on title and content similarity.

        Args:
            articles: List of articles to deduplicate

        Returns:
            List of unique articles
        """
        if not articles:
            return []

        unique_articles = []
        seen_hashes = set()

        for article in articles:
            # Create hash from normalized title and content
            normalized_title = self._normalize_text(article.title)
            normalized_content = self._normalize_text(
                article.content[:500]
            )  # First 500 chars

            content_hash = hashlib.md5(
                f"{normalized_title}|{normalized_content}".encode()
            ).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_articles.append(article)
            else:
                logger.debug(f"Duplicate article filtered: {article.title}")

        logger.info(
            f"Deduplicated {len(articles)} articles to {len(unique_articles)} unique articles"
        )
        return unique_articles

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by removing extra whitespace and punctuation.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r"\s+", " ", text.lower().strip())

        # Remove common punctuation that might vary
        normalized = re.sub(r"[^\w\s]", "", normalized)

        return normalized
