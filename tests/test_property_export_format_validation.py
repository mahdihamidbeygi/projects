"""
Simple property-based test for export format validation.

**Feature: news-market-predictor, Property 17: Export format validation**
"""

import json
import csv
from datetime import datetime
from io import StringIO

from hypothesis import given, strategies as st, settings

from news_market_predictor.models import (
    NewsArticle,
    export_to_json,
    export_to_csv,
)


@given(
    st.lists(
        st.builds(
            NewsArticle,
            id=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            title=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            content=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            url=st.just("https://example.com/test"),
            published_at=st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
            ),
            source=st.sampled_from(["Yahoo Finance", "Reuters"]),
            category=st.sampled_from(["Technology", "Finance"]),
            raw_metadata=st.dictionaries(
                keys=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
                values=st.text(max_size=20),
                min_size=0,
                max_size=2,
            ),
        ),
        min_size=1,
        max_size=3,
    )
)
@settings(max_examples=5)  # Reduced examples for faster execution
def test_export_format_validation_json(articles):
    """
    **Feature: news-market-predictor, Property 17: Export format validation**

    Property: For any exported results, the data should be valid JSON or CSV format
    and parseable by external tools.

    **Validates: Requirements 4.5**
    """
    # Export to JSON format
    json_output = export_to_json(articles)

    # Verify it's a valid JSON string
    assert isinstance(json_output, str)
    assert len(json_output) > 0

    # Verify it can be parsed by external JSON parser
    parsed_data = json.loads(json_output)

    # Verify structure is correct
    assert isinstance(parsed_data, list)
    assert len(parsed_data) == len(articles)

    # Verify each item has the expected fields
    for i, item in enumerate(parsed_data):
        assert isinstance(item, dict)
        assert "id" in item
        assert "title" in item
        assert "content" in item
        assert "url" in item
        assert "published_at" in item
        assert "source" in item
        assert "category" in item
        assert "raw_metadata" in item

        # Verify data integrity
        assert item["id"] == articles[i].id
        assert item["title"] == articles[i].title


@given(
    st.lists(
        st.builds(
            NewsArticle,
            id=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            title=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            content=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            url=st.just("https://example.com/test"),
            published_at=st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
            ),
            source=st.sampled_from(["Yahoo Finance", "Reuters"]),
            category=st.sampled_from(["Technology", "Finance"]),
            raw_metadata=st.dictionaries(
                keys=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
                values=st.text(max_size=20),
                min_size=0,
                max_size=2,
            ),
        ),
        min_size=1,
        max_size=3,
    )
)
@settings(max_examples=5)  # Reduced examples for faster execution
def test_export_format_validation_csv(articles):
    """
    **Feature: news-market-predictor, Property 17: Export format validation**

    Property: For any exported results, the data should be valid JSON or CSV format
    and parseable by external tools.

    **Validates: Requirements 4.5**
    """
    # Export to CSV format
    csv_output = export_to_csv(articles)

    # Verify it's a valid CSV string
    assert isinstance(csv_output, str)
    assert len(csv_output) > 0

    # Verify it can be parsed by external CSV parser
    csv_reader = csv.DictReader(StringIO(csv_output))
    rows = list(csv_reader)

    # Verify structure is correct
    assert len(rows) == len(articles)

    # Verify each row has the expected fields
    expected_fields = {
        "id",
        "title",
        "content",
        "url",
        "published_at",
        "source",
        "category",
        "raw_metadata",
    }
    for i, row in enumerate(rows):
        assert isinstance(row, dict)
        assert set(row.keys()) == expected_fields

        # Verify data integrity
        assert row["id"] == articles[i].id
        assert row["title"] == articles[i].title
        assert row["source"] == articles[i].source
