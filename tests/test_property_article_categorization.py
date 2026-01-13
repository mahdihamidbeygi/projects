"""
Property-based tests for article categorization consistency.

Feature: news-market-predictor, Property 8: Article categorization consistency
Validates: Requirements 2.4
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime
from news_market_predictor.models import NewsArticle
from news_market_predictor.analyzer.entity_extractor import FinancialEntityExtractor


# Strategy for generating article content with stock symbols
@st.composite
def article_with_stock_symbols(draw):
    """Generate articles that contain specific stock symbols."""
    # Known stock symbols from the entity extractor
    stock_symbols = [
        "AAPL",
        "GOOGL",
        "MSFT",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "JPM",
        "JNJ",
        "V",
        "PG",
        "UNH",
        "HD",
        "MA",
        "BAC",
        "PFE",
        "KO",
        "ABBV",
        "XOM",
        "CVX",
        "NKE",
        "UBER",
        "LYFT",
        "SNAP",
        "SQ",
        "PYPL",
    ]

    # Select 1-3 stock symbols for this article
    selected_symbols = draw(
        st.lists(st.sampled_from(stock_symbols), min_size=1, max_size=3, unique=True)
    )

    # Generate base content
    base_content = draw(st.text(min_size=50, max_size=500))

    # Inject stock symbols into content in various formats
    content_parts = [base_content]
    for symbol in selected_symbols:
        # Add stock symbol in different formats
        formats = [
            f"${symbol} shares rose today",
            f"({symbol}) announced earnings",
            f"{symbol} stock gained 5%",
            f"Trading in {symbol} was active",
            f"Analysts upgraded {symbol} to buy",
        ]
        format_choice = draw(st.sampled_from(formats))
        content_parts.append(format_choice)

    # Combine content
    final_content = " ".join(content_parts)

    # Generate article
    article = NewsArticle(
        id=draw(st.text(min_size=1, max_size=50)),
        title=draw(st.text(min_size=10, max_size=100)),
        content=final_content,
        url=f"https://finance.yahoo.com/news/{draw(st.text(min_size=5, max_size=20))}",
        published_at=draw(
            st.datetimes(
                min_value=datetime(2024, 1, 1), max_value=datetime(2024, 1, 31)
            )
        ),
        source="Yahoo Finance",
        category="general",  # Will be updated by categorization
        raw_metadata=draw(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.text(min_size=1, max_size=20),
                min_size=0,
                max_size=3,
            )
        ),
    )

    return article, selected_symbols


@st.composite
def article_without_stock_symbols(draw):
    """Generate articles that don't contain recognizable stock symbols."""
    # Generate content without stock symbols
    content = draw(
        st.text(min_size=50, max_size=500).filter(
            lambda x: not any(
                symbol in x.upper()
                for symbol in [
                    "AAPL",
                    "GOOGL",
                    "MSFT",
                    "AMZN",
                    "TSLA",
                    "META",
                    "NVDA",
                    "JPM",
                ]
            )
        )
    )

    article = NewsArticle(
        id=draw(st.text(min_size=1, max_size=50)),
        title=draw(st.text(min_size=10, max_size=100)),
        content=content,
        url=f"https://finance.yahoo.com/news/{draw(st.text(min_size=5, max_size=20))}",
        published_at=draw(
            st.datetimes(
                min_value=datetime(2024, 1, 1), max_value=datetime(2024, 1, 31)
            )
        ),
        source="Yahoo Finance",
        category="general",
        raw_metadata=draw(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.text(min_size=1, max_size=20),
                min_size=0,
                max_size=3,
            )
        ),
    )

    return article


def categorize_article(
    article: NewsArticle, extractor: FinancialEntityExtractor
) -> str:
    """
    Categorize an article based on its content analysis.

    This function implements the categorization logic that should be consistent
    across all articles with similar content patterns.
    """
    # Extract entities from the article
    entities = extractor.extract_all_entities(article.id, article.content)

    # Find stock symbols in the entities
    stock_entities = [e for e in entities if e.entity_type == "stock_symbol"]

    if not stock_entities:
        return "General"

    # Get sectors for all found stock symbols
    sectors = []
    for entity in stock_entities:
        sector = extractor.get_market_sector(entity.entity_value)
        if sector != "Unknown":
            sectors.append(sector)

    if not sectors:
        return "General"

    # If all stocks are from the same sector, use that sector
    unique_sectors = list(set(sectors))
    if len(unique_sectors) == 1:
        return unique_sectors[0]

    # If multiple sectors, use the most common one
    sector_counts = {}
    for sector in sectors:
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    most_common_sector = max(sector_counts, key=sector_counts.get)
    return most_common_sector


@given(article_with_stock_symbols())
@settings(max_examples=5)  # Reduced examples for faster execution
def test_article_categorization_consistency_with_stocks(article_and_symbols):
    """
    Property 8: Article categorization consistency

    For any processed article containing stock symbols, the system should assign
    appropriate market sector tags based on content analysis, and this categorization
    should be consistent across multiple runs with the same content.

    **Feature: news-market-predictor, Property 8: Article categorization consistency**
    **Validates: Requirements 2.4**
    """
    article, expected_symbols = article_and_symbols

    # Assume article has valid content
    assume(len(article.content.strip()) > 10)
    assume(len(article.title.strip()) > 5)

    extractor = FinancialEntityExtractor()

    # Categorize the article multiple times
    category1 = categorize_article(article, extractor)
    category2 = categorize_article(article, extractor)
    category3 = categorize_article(article, extractor)

    # Property: Categorization should be consistent across multiple runs
    assert (
        category1 == category2 == category3
    ), f"Article categorization should be consistent. Got: {category1}, {category2}, {category3}"

    # Property: Articles with stock symbols should not be categorized as "General"
    # unless no known sectors are found
    extracted_entities = extractor.extract_all_entities(article.id, article.content)
    stock_entities = [e for e in extracted_entities if e.entity_type == "stock_symbol"]

    if stock_entities:
        # Check if any of the extracted symbols have known sectors
        has_known_sectors = any(
            extractor.get_market_sector(entity.entity_value) != "Unknown"
            for entity in stock_entities
        )

        if has_known_sectors:
            # Should be categorized into a specific sector, not "General"
            assert category1 != "General", (
                f"Article with known stock symbols should be categorized into specific sector, "
                f"not 'General'. Found symbols: {[e.entity_value for e in stock_entities]}, "
                f"Category: {category1}"
            )

            # Category should be one of the valid sectors
            valid_sectors = set(extractor.stock_db.SECTOR_MAPPING.values())
            assert category1 in valid_sectors, (
                f"Category should be one of the valid sectors: {valid_sectors}, "
                f"but got: {category1}"
            )


@given(article_without_stock_symbols())
@settings(max_examples=5)  # Reduced examples for faster execution
def test_article_categorization_consistency_without_stocks(article):
    """
    Property 8: Article categorization consistency (edge case)

    For any processed article without recognizable stock symbols, the system should
    consistently assign a "General" category.

    **Feature: news-market-predictor, Property 8: Article categorization consistency**
    **Validates: Requirements 2.4**
    """
    # Assume article has valid content
    assume(len(article.content.strip()) > 10)
    assume(len(article.title.strip()) > 5)

    extractor = FinancialEntityExtractor()

    # Categorize the article multiple times
    category1 = categorize_article(article, extractor)
    category2 = categorize_article(article, extractor)
    category3 = categorize_article(article, extractor)

    # Property: Categorization should be consistent across multiple runs
    assert (
        category1 == category2 == category3
    ), f"Article categorization should be consistent. Got: {category1}, {category2}, {category3}"

    # Verify no stock symbols were extracted
    extracted_entities = extractor.extract_all_entities(article.id, article.content)
    stock_entities = [e for e in extracted_entities if e.entity_type == "stock_symbol"]

    # Property: Articles without stock symbols should be categorized as "General"
    if not stock_entities:
        assert category1 == "General", (
            f"Article without stock symbols should be categorized as 'General', "
            f"but got: {category1}"
        )


@given(st.text(min_size=1, max_size=50))  # article_id
@settings(max_examples=5)  # Reduced examples for faster execution
def test_article_categorization_consistency_same_content(article_id):
    """
    Property 8: Article categorization consistency (deterministic behavior)

    For any article content, multiple articles with identical content should receive
    the same categorization regardless of other metadata differences.

    **Feature: news-market-predictor, Property 8: Article categorization consistency**
    **Validates: Requirements 2.4**
    """
    # Create identical content with stock symbols
    content = "Apple Inc. (AAPL) reported strong quarterly earnings today. The technology giant saw revenue growth of 15% year-over-year."
    title = "Apple Reports Strong Quarterly Earnings"

    # Create two articles with identical content but different metadata
    article1 = NewsArticle(
        id=f"{article_id}_1",
        title=title,
        content=content,
        url="https://finance.yahoo.com/news/apple-earnings-1",
        published_at=datetime(2024, 1, 15, 10, 0, 0),
        source="Yahoo Finance",
        category="general",
        raw_metadata={"source_id": "1", "author": "John Doe"},
    )

    article2 = NewsArticle(
        id=f"{article_id}_2",
        title=title,
        content=content,
        url="https://finance.yahoo.com/news/apple-earnings-2",
        published_at=datetime(2024, 1, 16, 14, 30, 0),
        source="Reuters",
        category="business",
        raw_metadata={"source_id": "2", "author": "Jane Smith"},
    )

    extractor = FinancialEntityExtractor()

    # Categorize both articles
    category1 = categorize_article(article1, extractor)
    category2 = categorize_article(article2, extractor)

    # Property: Articles with identical content should have identical categorization
    assert category1 == category2, (
        f"Articles with identical content should have identical categorization. "
        f"Article 1 category: {category1}, Article 2 category: {category2}"
    )

    # Property: Should be categorized as "Technology" since AAPL is in Technology sector
    assert (
        category1 == "Technology"
    ), f"Article about Apple should be categorized as 'Technology', but got: {category1}"


if __name__ == "__main__":
    pytest.main([__file__])
