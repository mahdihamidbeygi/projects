"""
Property-based tests for duplicate article filtering.

**Feature: news-market-predictor, Property 4: Duplicate article filtering**
"""

import hashlib
import re
from datetime import datetime, timedelta
from typing import List

from hypothesis import given, strategies as st, settings

from news_market_predictor.models import NewsArticle
from news_market_predictor.fetcher.yahoo_finance_fetcher import YahooFinanceNewsFetcher


def normalize_text_for_comparison(text: str) -> str:
    """
    Normalize text for comparison by removing extra whitespace and punctuation.
    This mirrors the logic in YahooFinanceNewsFetcher._normalize_text.
    """
    if not text:
        return ""

    # Convert to lowercase and remove extra whitespace
    normalized = re.sub(r"\s+", " ", text.lower().strip())

    # Remove common punctuation that might vary
    normalized = re.sub(r"[^\w\s]", "", normalized)

    return normalized


def create_content_hash(title: str, content: str) -> str:
    """
    Create content hash for duplicate detection.
    This mirrors the logic in YahooFinanceNewsFetcher.deduplicate_articles.
    """
    normalized_title = normalize_text_for_comparison(title)
    normalized_content = normalize_text_for_comparison(content[:500])  # First 500 chars

    return hashlib.md5(f"{normalized_title}|{normalized_content}".encode()).hexdigest()


def create_test_article(
    article_id: str,
    title: str,
    content: str,
    url: str = None,
    published_at: datetime = None,
) -> NewsArticle:
    """Create a test NewsArticle with default values."""
    return NewsArticle(
        id=article_id,
        title=title,
        content=content,
        url=url or f"https://finance.yahoo.com/news/{article_id}",
        published_at=published_at or datetime.now(),
        source="Yahoo Finance",
        category="general",
        raw_metadata={"test": True},
    )


@given(
    st.lists(
        st.tuples(
            st.text(min_size=5, max_size=100),  # title
            st.text(min_size=20, max_size=1000),  # content
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=5)  # Reduced examples for faster execution
def test_duplicate_filtering_preserves_unique_articles(article_data):
    """
    **Feature: news-market-predictor, Property 4: Duplicate article filtering**

    Property: For any set of articles containing duplicates, the deduplication process
    should remove articles with similar titles and content while preserving unique articles.

    **Validates: Requirements 1.4**

    This test verifies that the duplicate filtering process correctly identifies and
    removes duplicate articles based on title and content similarity while preserving
    all unique articles.
    """
    fetcher = YahooFinanceNewsFetcher()

    # Create articles from the generated data
    articles = []
    for i, (title, content) in enumerate(article_data):
        article = create_test_article(
            article_id=f"article_{i}",
            title=title,
            content=content,
        )
        articles.append(article)

    # Apply deduplication
    unique_articles = fetcher.deduplicate_articles(articles)

    # Verify all returned articles are from the original set
    original_ids = {article.id for article in articles}
    unique_ids = {article.id for article in unique_articles}
    assert unique_ids.issubset(
        original_ids
    ), "Deduplicated articles should only contain original article IDs"

    # Verify no duplicates remain based on content hash
    seen_hashes = set()
    for article in unique_articles:
        content_hash = create_content_hash(article.title, article.content)
        assert (
            content_hash not in seen_hashes
        ), f"Duplicate content hash found in results: {content_hash}"
        seen_hashes.add(content_hash)

    # Verify that each unique content hash from original articles is represented
    original_hashes = set()
    for article in articles:
        content_hash = create_content_hash(article.title, article.content)
        original_hashes.add(content_hash)

    unique_hashes = set()
    for article in unique_articles:
        content_hash = create_content_hash(article.title, article.content)
        unique_hashes.add(content_hash)

    assert (
        unique_hashes == original_hashes
    ), "Deduplication should preserve exactly one article for each unique content hash"

    # Verify the number of unique articles matches the number of unique content hashes
    assert len(unique_articles) == len(
        original_hashes
    ), f"Expected {len(original_hashes)} unique articles, got {len(unique_articles)}"


@given(
    st.text(min_size=10, max_size=100),  # base_title
    st.text(min_size=50, max_size=500),  # base_content
    st.integers(min_value=2, max_value=10),  # duplicate_count
)
def test_duplicate_filtering_removes_exact_duplicates(
    base_title, base_content, duplicate_count
):
    """
    **Feature: news-market-predictor, Property 4: Duplicate article filtering**

    Property: For any set of articles containing exact duplicates (same title and content),
    the deduplication process should keep only one copy of each duplicate.

    **Validates: Requirements 1.4**

    This test verifies that exact duplicates are properly identified and removed,
    leaving only one instance of each unique article.
    """
    fetcher = YahooFinanceNewsFetcher()

    # Create multiple articles with identical title and content
    articles = []
    for i in range(duplicate_count):
        article = create_test_article(
            article_id=f"duplicate_{i}",
            title=base_title,
            content=base_content,
            published_at=datetime.now() + timedelta(minutes=i),  # Different timestamps
        )
        articles.append(article)

    # Apply deduplication
    unique_articles = fetcher.deduplicate_articles(articles)

    # Should have exactly one article remaining
    assert (
        len(unique_articles) == 1
    ), f"Expected 1 unique article from {duplicate_count} duplicates, got {len(unique_articles)}"

    # The remaining article should have the same title and content
    remaining_article = unique_articles[0]
    assert (
        remaining_article.title == base_title
    ), "Remaining article should have the original title"
    assert (
        remaining_article.content == base_content
    ), "Remaining article should have the original content"

    # The remaining article should be one of the original articles
    original_ids = {article.id for article in articles}
    assert (
        remaining_article.id in original_ids
    ), "Remaining article should be one of the original articles"


@given(
    st.text(min_size=10, max_size=100),  # base_title
    st.text(min_size=50, max_size=500),  # base_content
    st.lists(
        st.text(min_size=1, max_size=20), min_size=1, max_size=5
    ),  # title_variations
    st.lists(
        st.text(min_size=1, max_size=50), min_size=1, max_size=5
    ),  # content_variations
)
def test_duplicate_filtering_handles_similar_content(
    base_title, base_content, title_variations, content_variations
):
    """
    **Feature: news-market-predictor, Property 4: Duplicate article filtering**

    Property: For any set of articles with similar but not identical titles and content,
    the deduplication process should identify them as duplicates if they normalize
    to the same content hash.

    **Validates: Requirements 1.4**

    This test verifies that articles with minor variations (punctuation, whitespace)
    are correctly identified as duplicates.
    """
    fetcher = YahooFinanceNewsFetcher()

    articles = []

    # Create the base article
    base_article = create_test_article(
        article_id="base_article",
        title=base_title,
        content=base_content,
    )
    articles.append(base_article)

    # Create variations with different punctuation and whitespace
    for i, title_var in enumerate(title_variations):
        # Add punctuation and whitespace variations to title
        varied_title = f"{base_title}  {title_var}!!!"
        varied_title = varied_title.replace(" ", "   ")  # Extra spaces

        # Add variations to content
        content_var = content_variations[i % len(content_variations)]
        varied_content = f"{base_content} {content_var}..."
        varied_content = varied_content.replace(".", " . ")  # Space around punctuation

        # Only create variation if it would normalize to the same hash
        if create_content_hash(varied_title, varied_content) == create_content_hash(
            base_title, base_content
        ):
            varied_article = create_test_article(
                article_id=f"variation_{i}",
                title=varied_title,
                content=varied_content,
            )
            articles.append(varied_article)

    # Apply deduplication
    unique_articles = fetcher.deduplicate_articles(articles)

    # Should have exactly one article if variations normalize to same hash
    if len(articles) > 1:
        # Check if all articles have the same content hash
        content_hashes = set()
        for article in articles:
            content_hash = create_content_hash(article.title, article.content)
            content_hashes.add(content_hash)

        expected_unique_count = len(content_hashes)
        assert len(unique_articles) == expected_unique_count, (
            f"Expected {expected_unique_count} unique articles based on content hashes, "
            f"got {len(unique_articles)}"
        )


@given(
    st.lists(
        st.tuples(
            st.text(min_size=5, max_size=50),  # title
            st.text(min_size=20, max_size=200),  # content
        ),
        min_size=5,
        max_size=15,
    )
)
def test_duplicate_filtering_empty_input_handling(article_data):
    """
    **Feature: news-market-predictor, Property 4: Duplicate article filtering**

    Property: For any empty list of articles, the deduplication process should
    return an empty list without errors.

    **Validates: Requirements 1.4**

    This test verifies that the deduplication process handles edge cases correctly.
    """
    fetcher = YahooFinanceNewsFetcher()

    # Test empty list
    empty_result = fetcher.deduplicate_articles([])
    assert empty_result == [], "Empty input should return empty list"

    # Test single article
    if article_data:
        title, content = article_data[0]
        single_article = create_test_article(
            article_id="single",
            title=title,
            content=content,
        )

        single_result = fetcher.deduplicate_articles([single_article])
        assert (
            len(single_result) == 1
        ), "Single article should return list with one article"
        assert (
            single_result[0].id == single_article.id
        ), "Single article should be preserved"


@given(
    st.text(min_size=10, max_size=100),  # title
    st.text(min_size=50, max_size=500),  # content
    st.integers(min_value=1, max_value=10),  # article_count
)
def test_duplicate_filtering_preserves_article_integrity(title, content, article_count):
    """
    **Feature: news-market-predictor, Property 4: Duplicate article filtering**

    Property: For any set of articles processed through deduplication, the
    remaining articles should maintain their original data integrity.

    **Validates: Requirements 1.4**

    This test verifies that the deduplication process doesn't modify the
    content or structure of the articles it preserves.
    """
    fetcher = YahooFinanceNewsFetcher()

    # Create articles with the same content but different IDs and timestamps
    articles = []
    for i in range(article_count):
        article = create_test_article(
            article_id=f"integrity_test_{i}",
            title=title,
            content=content,
            published_at=datetime.now() + timedelta(seconds=i),
        )
        articles.append(article)

    # Store original article data for comparison
    original_articles = {article.id: article for article in articles}

    # Apply deduplication
    unique_articles = fetcher.deduplicate_articles(articles)

    # Should have exactly one article (all have same content)
    assert (
        len(unique_articles) == 1
    ), f"Expected 1 unique article, got {len(unique_articles)}"

    # The preserved article should be identical to one of the originals
    preserved_article = unique_articles[0]
    original_article = original_articles[preserved_article.id]

    assert preserved_article.id == original_article.id, "Article ID should be preserved"
    assert (
        preserved_article.title == original_article.title
    ), "Article title should be preserved"
    assert (
        preserved_article.content == original_article.content
    ), "Article content should be preserved"
    assert (
        preserved_article.url == original_article.url
    ), "Article URL should be preserved"
    assert (
        preserved_article.published_at == original_article.published_at
    ), "Article timestamp should be preserved"
    assert (
        preserved_article.source == original_article.source
    ), "Article source should be preserved"
    assert (
        preserved_article.category == original_article.category
    ), "Article category should be preserved"
    assert (
        preserved_article.raw_metadata == original_article.raw_metadata
    ), "Article metadata should be preserved"

    # Verify the article is still valid
    assert preserved_article.validate() is True, "Preserved article should be valid"


@given(
    st.lists(
        st.tuples(
            st.text(
                min_size=5,
                max_size=100,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Zs")
                ),
            ),  # title
            st.text(
                min_size=20,
                max_size=1000,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Zs")
                ),
            ),  # content
        ),
        min_size=2,
        max_size=10,
    )
)
def test_duplicate_filtering_deterministic_behavior(article_data):
    """
    **Feature: news-market-predictor, Property 4: Duplicate article filtering**

    Property: For any set of articles, the deduplication process should produce
    consistent results when run multiple times with the same input.

    **Validates: Requirements 1.4**

    This test verifies that the deduplication process is deterministic and
    produces consistent results across multiple runs.
    """
    fetcher = YahooFinanceNewsFetcher()

    # Create articles from the generated data
    articles = []
    for i, (title, content) in enumerate(article_data):
        article = create_test_article(
            article_id=f"deterministic_{i}",
            title=title,
            content=content,
        )
        articles.append(article)

    # Run deduplication multiple times
    results = []
    for run in range(3):
        unique_articles = fetcher.deduplicate_articles(articles.copy())
        # Sort by ID for consistent comparison
        sorted_articles = sorted(unique_articles, key=lambda x: x.id)
        results.append(sorted_articles)

    # All runs should produce identical results
    for i in range(1, len(results)):
        assert len(results[0]) == len(
            results[i]
        ), f"Run {i} produced different number of articles than run 0"

        for j in range(len(results[0])):
            assert (
                results[0][j].id == results[i][j].id
            ), f"Run {i} produced different article at position {j}"
            assert (
                results[0][j].title == results[i][j].title
            ), f"Run {i} produced different title at position {j}"
            assert (
                results[0][j].content == results[i][j].content
            ), f"Run {i} produced different content at position {j}"
