"""
Property-based tests for article storage format consistency.

**Feature: news-market-predictor, Property 5: Article storage format consistency**
"""

import json
import tempfile
import os
from datetime import datetime
from typing import Dict, Any

from hypothesis import given, strategies as st

from news_market_predictor.models import NewsArticle


def serialize_article_to_json(article: NewsArticle) -> str:
    """Serialize NewsArticle to JSON format."""
    article_dict = {
        "id": article.id,
        "title": article.title,
        "content": article.content,
        "url": article.url,
        "published_at": article.published_at.isoformat(),
        "source": article.source,
        "category": article.category,
        "raw_metadata": article.raw_metadata,
    }
    return json.dumps(article_dict, sort_keys=True)


def deserialize_article_from_json(json_str: str) -> NewsArticle:
    """Deserialize NewsArticle from JSON format."""
    data = json.loads(json_str)
    return NewsArticle(
        id=data["id"],
        title=data["title"],
        content=data["content"],
        url=data["url"],
        published_at=datetime.fromisoformat(data["published_at"]),
        source=data["source"],
        category=data["category"],
        raw_metadata=data["raw_metadata"],
    )


def serialize_article_to_csv_row(article: NewsArticle) -> Dict[str, Any]:
    """Serialize NewsArticle to CSV-compatible dictionary format."""
    return {
        "id": article.id,
        "title": article.title,
        "content": article.content,
        "url": article.url,
        "published_at": article.published_at.isoformat(),
        "source": article.source,
        "category": article.category,
        "raw_metadata": json.dumps(article.raw_metadata, sort_keys=True),
    }


def deserialize_article_from_csv_row(row_dict: Dict[str, Any]) -> NewsArticle:
    """Deserialize NewsArticle from CSV row dictionary format."""
    return NewsArticle(
        id=row_dict["id"],
        title=row_dict["title"],
        content=row_dict["content"],
        url=row_dict["url"],
        published_at=datetime.fromisoformat(row_dict["published_at"]),
        source=row_dict["source"],
        category=row_dict["category"],
        raw_metadata=json.loads(row_dict["raw_metadata"]),
    )


def store_and_retrieve_json_file(article: NewsArticle) -> NewsArticle:
    """Store article to JSON file and retrieve it back."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_data = serialize_article_to_json(article)
        f.write(json_data)
        temp_path = f.name

    try:
        with open(temp_path, "r") as f:
            retrieved_json = f.read()
        return deserialize_article_from_json(retrieved_json)
    finally:
        os.unlink(temp_path)


@given(
    st.text(min_size=1, max_size=50),  # id
    st.text(min_size=1, max_size=200),  # title
    st.text(min_size=1, max_size=1000),  # content
    st.text(min_size=10, max_size=100),  # url_part
    st.datetimes(
        min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
    ),  # published_at
    st.sampled_from(["Yahoo Finance", "Reuters", "Bloomberg", "MarketWatch"]),  # source
    st.sampled_from(
        ["Technology", "Finance", "Healthcare", "Energy", "Consumer"]
    ),  # category
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
        min_size=0,
        max_size=5,
    ),  # raw_metadata
)
def test_article_storage_format_consistency_json_round_trip(
    id_val, title, content, url_part, published_at, source, category, raw_metadata
):
    """
    **Feature: news-market-predictor, Property 5: Article storage format consistency**

    Property: For any article processed and stored, retrieving it should return
    the same structured format with all original data intact.

    **Validates: Requirements 1.5**

    This test verifies that storing an article in JSON format and then retrieving
    it preserves all original data without loss or corruption.
    """
    # Create the article
    article = NewsArticle(
        id=id_val,
        title=title,
        content=content,
        url=f"https://example.com/{url_part}",
        published_at=published_at,
        source=source,
        category=category,
        raw_metadata=raw_metadata,
    )

    # Store and retrieve the article through JSON file operations
    retrieved_article = store_and_retrieve_json_file(article)

    # Verify all fields are preserved exactly
    assert retrieved_article.id == article.id
    assert retrieved_article.title == article.title
    assert retrieved_article.content == article.content
    assert retrieved_article.url == article.url
    assert retrieved_article.published_at == article.published_at
    assert retrieved_article.source == article.source
    assert retrieved_article.category == article.category
    assert retrieved_article.raw_metadata == article.raw_metadata


@given(
    st.text(min_size=1, max_size=50),  # id
    st.text(min_size=1, max_size=200),  # title
    st.text(min_size=1, max_size=1000),  # content
    st.text(min_size=10, max_size=100),  # url_part
    st.datetimes(
        min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
    ),  # published_at
    st.sampled_from(["Yahoo Finance", "Reuters", "Bloomberg", "MarketWatch"]),  # source
    st.sampled_from(
        ["Technology", "Finance", "Healthcare", "Energy", "Consumer"]
    ),  # category
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
        min_size=0,
        max_size=5,
    ),  # raw_metadata
)
def test_article_storage_format_consistency_csv_round_trip(
    id_val, title, content, url_part, published_at, source, category, raw_metadata
):
    """
    **Feature: news-market-predictor, Property 5: Article storage format consistency**

    Property: For any article processed and stored, retrieving it should return
    the same structured format with all original data intact.

    **Validates: Requirements 1.5**

    This test verifies that storing an article in CSV format and then retrieving
    it preserves all original data without loss or corruption.
    """
    # Create the article
    article = NewsArticle(
        id=id_val,
        title=title,
        content=content,
        url=f"https://example.com/{url_part}",
        published_at=published_at,
        source=source,
        category=category,
        raw_metadata=raw_metadata,
    )

    # Serialize to CSV format and back
    csv_row = serialize_article_to_csv_row(article)
    retrieved_article = deserialize_article_from_csv_row(csv_row)

    # Verify all fields are preserved exactly
    assert retrieved_article.id == article.id
    assert retrieved_article.title == article.title
    assert retrieved_article.content == article.content
    assert retrieved_article.url == article.url
    assert retrieved_article.published_at == article.published_at
    assert retrieved_article.source == article.source
    assert retrieved_article.category == article.category
    assert retrieved_article.raw_metadata == article.raw_metadata


@given(
    st.text(min_size=1, max_size=50),  # id
    st.text(min_size=1, max_size=200),  # title
    st.text(min_size=1, max_size=1000),  # content
    st.text(min_size=10, max_size=100),  # url_part
    st.datetimes(
        min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
    ),  # published_at
    st.sampled_from(["Yahoo Finance", "Reuters", "Bloomberg", "MarketWatch"]),  # source
    st.sampled_from(
        ["Technology", "Finance", "Healthcare", "Energy", "Consumer"]
    ),  # category
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
        min_size=0,
        max_size=5,
    ),  # raw_metadata
)
def test_article_storage_format_consistency_direct_serialization(
    id_val, title, content, url_part, published_at, source, category, raw_metadata
):
    """
    **Feature: news-market-predictor, Property 5: Article storage format consistency**

    Property: For any article processed and stored, retrieving it should return
    the same structured format with all original data intact.

    **Validates: Requirements 1.5**

    This test verifies that direct serialization/deserialization preserves
    all original data without loss or corruption.
    """
    # Create the article
    article = NewsArticle(
        id=id_val,
        title=title,
        content=content,
        url=f"https://example.com/{url_part}",
        published_at=published_at,
        source=source,
        category=category,
        raw_metadata=raw_metadata,
    )

    # Test direct JSON serialization round-trip
    json_str = serialize_article_to_json(article)
    retrieved_article = deserialize_article_from_json(json_str)

    # Verify all fields are preserved exactly
    assert retrieved_article.id == article.id
    assert retrieved_article.title == article.title
    assert retrieved_article.content == article.content
    assert retrieved_article.url == article.url
    assert retrieved_article.published_at == article.published_at
    assert retrieved_article.source == article.source
    assert retrieved_article.category == article.category
    assert retrieved_article.raw_metadata == article.raw_metadata
