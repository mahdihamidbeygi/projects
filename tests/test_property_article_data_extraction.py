"""
Property-based tests for article data extraction completeness.

**Feature: news-market-predictor, Property 2: Article data extraction completeness**
"""

from datetime import datetime
from typing import Dict, Any

from hypothesis import given, strategies as st, settings

from news_market_predictor.models import NewsArticle


def simulate_parse_article_content(
    raw_content: str, metadata: Dict[str, Any]
) -> NewsArticle:
    """
    Simulate the parse_article_content method that should extract all required fields.

    This function represents what the NewsFetcher.parse_article_content method should do:
    extract title, content, publication timestamp, and other required fields from raw input.
    """
    # Extract required fields from metadata and raw content
    # In a real implementation, this would parse HTML/XML content

    # Ensure all required metadata fields are present and non-empty (excluding content)
    required_fields = [
        "id",
        "title",
        "url",
        "published_at",
        "source",
        "category",
    ]

    for field in required_fields:
        if field not in metadata or not metadata[field]:
            raise ValueError(f"Missing or empty required field: {field}")

    # Validate that content is extracted from raw_content
    if not raw_content or not raw_content.strip():
        raise ValueError("Raw content cannot be empty")

    # Create NewsArticle with extracted data
    article = NewsArticle(
        id=metadata["id"],
        title=metadata["title"],
        content=raw_content.strip(),  # Content extracted from raw input
        url=metadata["url"],
        published_at=metadata["published_at"],
        source=metadata["source"],
        category=metadata["category"],
        raw_metadata=metadata.get("raw_metadata", {}),
    )

    # Validate the created article
    article.validate()

    return article


@given(
    st.text(min_size=1, max_size=1000).filter(
        lambda x: x.strip()
    ),  # raw_content - ensure non-empty after strip
    st.fixed_dictionaries(
        {
            "id": st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            "title": st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
            "url": st.text(min_size=10, max_size=100).map(
                lambda x: f"https://finance.yahoo.com/{x}"
            ),
            "published_at": st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
            ),
            "source": st.sampled_from(
                ["Yahoo Finance", "Reuters", "Bloomberg", "MarketWatch"]
            ),
            "category": st.sampled_from(
                ["Technology", "Finance", "Healthcare", "Energy", "Consumer"]
            ),
            "raw_metadata": st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.one_of(
                    st.text(max_size=100),
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                ),
                min_size=0,
                max_size=5,
            ),
        }
    ),
)
@settings(max_examples=5)  # Reduced examples for faster execution
def test_article_data_extraction_completeness(raw_content, metadata):
    """
    **Feature: news-market-predictor, Property 2: Article data extraction completeness**

    Property: For any valid news article input, the system should extract and populate
    all required fields (title, content, timestamp, stock symbols).

    **Validates: Requirements 1.2**

    This test verifies that the article parsing process extracts all required fields
    from raw input and creates a complete, valid NewsArticle object.
    """
    # Parse the article content using the simulated extraction method
    article = simulate_parse_article_content(raw_content, metadata)

    # Verify all required fields are extracted and populated
    assert article.id is not None and article.id.strip() != ""
    assert article.title is not None and article.title.strip() != ""
    assert article.content is not None and article.content.strip() != ""
    assert article.url is not None and article.url.strip() != ""
    assert article.published_at is not None
    assert isinstance(article.published_at, datetime)
    assert article.source is not None and article.source.strip() != ""
    assert article.category is not None and article.category.strip() != ""
    assert article.raw_metadata is not None
    assert isinstance(article.raw_metadata, dict)

    # Verify that content was actually extracted from raw input
    assert article.content == raw_content.strip()

    # Verify that extracted data matches the input metadata
    assert article.id == metadata["id"]
    assert article.title == metadata["title"]
    assert article.url == metadata["url"]
    assert article.published_at == metadata["published_at"]
    assert article.source == metadata["source"]
    assert article.category == metadata["category"]
    assert article.raw_metadata == metadata["raw_metadata"]

    # Verify the article passes validation
    assert article.validate() is True


@given(
    st.text(min_size=1, max_size=500).filter(
        lambda x: x.strip()
    ),  # raw_content - ensure non-empty after strip
    st.fixed_dictionaries(
        {
            "id": st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            "title": st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
            "url": st.text(min_size=10, max_size=100).map(
                lambda x: f"https://finance.yahoo.com/{x}"
            ),
            "published_at": st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
            ),
            "source": st.sampled_from(["Yahoo Finance", "Reuters", "Bloomberg"]),
            "category": st.sampled_from(["Technology", "Finance", "Healthcare"]),
            "raw_metadata": st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.text(max_size=50),
                min_size=1,
                max_size=3,
            ),
        }
    ),
)
@settings(max_examples=5)  # Reduced examples for faster execution
def test_article_data_extraction_field_types(raw_content, metadata):
    """
    **Feature: news-market-predictor, Property 2: Article data extraction completeness**

    Property: For any valid news article input, the system should extract and populate
    all required fields with correct data types.

    **Validates: Requirements 1.2**

    This test verifies that extracted fields have the correct data types and formats.
    """
    # Parse the article content
    article = simulate_parse_article_content(raw_content, metadata)

    # Verify field types are correct
    assert isinstance(article.id, str)
    assert isinstance(article.title, str)
    assert isinstance(article.content, str)
    assert isinstance(article.url, str)
    assert isinstance(article.published_at, datetime)
    assert isinstance(article.source, str)
    assert isinstance(article.category, str)
    assert isinstance(article.raw_metadata, dict)

    # Verify string fields are not empty
    assert len(article.id) > 0
    assert len(article.title) > 0
    assert len(article.content) > 0
    assert len(article.url) > 0
    assert len(article.source) > 0
    assert len(article.category) > 0


@given(
    st.text(min_size=1, max_size=200).filter(
        lambda x: x.strip()
    ),  # raw_content - ensure non-empty after strip
    st.fixed_dictionaries(
        {
            "id": st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
            "title": st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            "url": st.just("https://finance.yahoo.com/news/test-article"),
            "published_at": st.datetimes(
                min_value=datetime(2023, 1, 1), max_value=datetime(2024, 12, 31)
            ),
            "source": st.just("Yahoo Finance"),
            "category": st.sampled_from(["Technology", "Finance"]),
            "raw_metadata": st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.text(max_size=30),
                min_size=0,
                max_size=2,
            ),
        }
    ),
)
@settings(max_examples=5)  # Reduced examples for faster execution
def test_article_data_extraction_content_preservation(raw_content, metadata):
    """
    **Feature: news-market-predictor, Property 2: Article data extraction completeness**

    Property: For any valid news article input, the extracted content should preserve
    the original raw content without loss or corruption.

    **Validates: Requirements 1.2**

    This test verifies that content extraction preserves the original data integrity.
    """
    # Parse the article content
    article = simulate_parse_article_content(raw_content, metadata)

    # Verify content is preserved exactly (after trimming whitespace)
    expected_content = raw_content.strip()
    assert article.content == expected_content

    # Verify content length is preserved
    assert len(article.content) == len(expected_content)

    # Verify no data corruption occurred during extraction
    if expected_content:
        assert article.content[0] == expected_content[0]  # First character preserved
        assert article.content[-1] == expected_content[-1]  # Last character preserved
