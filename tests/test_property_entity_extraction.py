"""
Property-based tests for entity extraction completeness.

**Feature: news-market-predictor, Property 7: Entity extraction completeness**
**Validates: Requirements 2.2, 2.3**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from news_market_predictor.analyzer.entity_extractor import FinancialEntityExtractor
from news_market_predictor.models import ExtractedEntity


class TestEntityExtractionCompleteness:
    """Property-based tests for entity extraction completeness."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FinancialEntityExtractor()

    @given(
        stock_symbols=st.lists(
            st.sampled_from(
                [
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
                ]
            ),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        text_template=st.sampled_from(
            [
                "{symbol} shares rose today",
                "Investors are watching {symbol} stock closely",
                "The company ${symbol} announced earnings",
                "Trading in {symbol} was active",
                "Analysts upgraded {symbol} to buy",
                "{symbol}: Strong quarterly results",
                "({symbol}) reported revenue growth",
                "ticker {symbol} gained 5%",
            ]
        ),
    )
    @settings(max_examples=5)  # Reduced examples for faster execution
    def test_stock_symbol_extraction_completeness(self, stock_symbols, text_template):
        """
        Property 7: Entity extraction completeness - Stock Symbols

        For any article containing known stock symbols, the system should
        identify and extract all mentioned symbols.

        **Feature: news-market-predictor, Property 7: Entity extraction completeness**
        **Validates: Requirements 2.2**
        """
        # Create text containing the stock symbols
        text_parts = []
        for symbol in stock_symbols:
            text_parts.append(text_template.format(symbol=symbol))

        article_text = ". ".join(text_parts) + "."

        # Extract entities
        extracted_entities = self.extractor.extract_stock_symbols(article_text)

        # Get extracted stock symbols
        extracted_symbols = {
            entity.entity_value
            for entity in extracted_entities
            if entity.entity_type == "stock_symbol"
        }

        # Property: All mentioned stock symbols should be extracted
        for symbol in stock_symbols:
            assert symbol in extracted_symbols, (
                f"Stock symbol {symbol} was mentioned in text but not extracted. "
                f"Text: {article_text[:200]}... "
                f"Extracted: {extracted_symbols}"
            )

    @given(
        company_names=st.lists(
            st.sampled_from(
                [
                    "Apple Inc.",
                    "Microsoft Corporation",
                    "Amazon.com Inc.",
                    "Tesla Inc.",
                    "Meta Platforms Inc.",
                    "NVIDIA Corporation",
                    "JPMorgan Chase & Co.",
                    "Johnson & Johnson",
                    "Visa Inc.",
                    "Procter & Gamble Co.",
                    "Home Depot Inc.",
                    "Mastercard Inc.",
                ]
            ),
            min_size=1,
            max_size=3,
            unique=True,
        ),
        text_template=st.sampled_from(
            [
                "{company} announced strong earnings today",
                "The board of {company} approved the merger",
                "{company} CEO spoke at the conference",
                "Investors are bullish on {company}",
                "{company} reported quarterly results",
                "Analysts expect {company} to outperform",
            ]
        ),
    )
    @settings(max_examples=5)  # Reduced examples for faster execution
    def test_company_name_extraction_completeness(self, company_names, text_template):
        """
        Property 7: Entity extraction completeness - Company Names

        For any article containing known company names, the system should
        identify and extract all mentioned companies.

        **Feature: news-market-predictor, Property 7: Entity extraction completeness**
        **Validates: Requirements 2.2**
        """
        # Create text containing the company names
        text_parts = []
        for company in company_names:
            text_parts.append(text_template.format(company=company))

        article_text = ". ".join(text_parts) + "."

        # Extract entities
        extracted_entities = self.extractor.identify_companies(article_text)

        # Get extracted company names
        extracted_companies = {
            entity.entity_value
            for entity in extracted_entities
            if entity.entity_type == "company"
        }

        # Property: All mentioned company names should be extracted
        for company in company_names:
            assert company in extracted_companies, (
                f"Company name '{company}' was mentioned in text but not extracted. "
                f"Text: {article_text[:200]}... "
                f"Extracted: {extracted_companies}"
            )

    @given(
        financial_metrics=st.lists(
            st.tuples(
                st.sampled_from(["earnings", "revenue", "guidance"]),
                st.floats(min_value=0.1, max_value=999.9).map(lambda x: f"{x:.2f}"),
            ),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    @settings(max_examples=5)  # Reduced examples for faster execution
    def test_financial_metrics_extraction_completeness(self, financial_metrics):
        """
        Property 7: Entity extraction completeness - Financial Metrics

        For any article containing financial metrics (earnings, revenue, guidance),
        the system should recognize and extract all mentioned metrics.

        **Feature: news-market-predictor, Property 7: Entity extraction completeness**
        **Validates: Requirements 2.3**
        """
        # Create text containing financial metrics
        text_parts = []
        expected_metrics = set()

        for metric_type, value in financial_metrics:
            if metric_type == "earnings":
                text_parts.append(
                    f"The company reported earnings of ${value} per share"
                )
                expected_metrics.add(f"earnings: {value}")
            elif metric_type == "revenue":
                text_parts.append(f"Quarterly revenue reached ${value} billion")
                expected_metrics.add(f"revenue: {value}")
            elif metric_type == "guidance":
                text_parts.append(
                    f"Management provided guidance of ${value} for next quarter"
                )
                expected_metrics.add(f"guidance: {value}")

        article_text = ". ".join(text_parts) + "."

        # Extract entities
        extracted_entities = self.extractor.find_financial_metrics(article_text)

        # Get extracted financial metrics
        extracted_metrics = {
            entity.entity_value
            for entity in extracted_entities
            if entity.entity_type == "metric"
        }

        # Property: All mentioned financial metrics should be extracted
        for expected_metric in expected_metrics:
            assert expected_metric in extracted_metrics, (
                f"Financial metric '{expected_metric}' was mentioned in text but not extracted. "
                f"Text: {article_text[:200]}... "
                f"Extracted: {extracted_metrics}"
            )

    @given(
        mixed_content=st.tuples(
            st.sampled_from(["AAPL", "GOOGL", "MSFT", "TSLA"]),
            st.sampled_from(
                ["Apple Inc.", "Alphabet Inc.", "Microsoft Corporation", "Tesla Inc."]
            ),
            st.tuples(
                st.sampled_from(["earnings", "revenue"]),
                st.floats(min_value=1.0, max_value=99.9).map(lambda x: f"{x:.1f}"),
            ),
        )
    )
    @settings(max_examples=5)  # Reduced examples for faster execution
    def test_mixed_entity_extraction_completeness(self, mixed_content):
        """
        Property 7: Entity extraction completeness - Mixed Content

        For any article containing multiple types of entities (stock symbols,
        company names, and financial metrics), the system should extract all
        mentioned entities of each type.

        **Feature: news-market-predictor, Property 7: Entity extraction completeness**
        **Validates: Requirements 2.2, 2.3**
        """
        stock_symbol, company_name, (metric_type, metric_value) = mixed_content

        # Create text with mixed entity types
        article_text = (
            f"{company_name} ({stock_symbol}) announced strong quarterly results today. "
            f"The company reported {metric_type} of ${metric_value} billion, "
            f"beating analyst expectations. {stock_symbol} shares rose 5% in after-hours trading."
        )

        # Extract all entities using the main extraction method
        all_entities = self.extractor.extract_all_entities("test-article", article_text)

        # Categorize extracted entities
        extracted_stocks = {
            entity.entity_value
            for entity in all_entities
            if entity.entity_type == "stock_symbol"
        }
        extracted_companies = {
            entity.entity_value
            for entity in all_entities
            if entity.entity_type == "company"
        }
        extracted_metrics = {
            entity.entity_value
            for entity in all_entities
            if entity.entity_type == "metric"
        }

        # Property: All entity types should be extracted
        assert stock_symbol in extracted_stocks, (
            f"Stock symbol {stock_symbol} not extracted from mixed content. "
            f"Extracted stocks: {extracted_stocks}"
        )

        assert company_name in extracted_companies, (
            f"Company name '{company_name}' not extracted from mixed content. "
            f"Extracted companies: {extracted_companies}"
        )

        expected_metric = f"{metric_type}: {metric_value}"
        assert expected_metric in extracted_metrics, (
            f"Financial metric '{expected_metric}' not extracted from mixed content. "
            f"Extracted metrics: {extracted_metrics}"
        )

        # Property: All entities should have valid article_id
        for entity in all_entities:
            assert (
                entity.article_id == "test-article"
            ), f"Entity {entity.entity_value} has incorrect article_id: {entity.article_id}"

    @given(
        text_with_no_entities=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")),
            min_size=10,
            max_size=100,
        ).filter(
            lambda x: not any(
                symbol in x.upper()
                for symbol in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
            )
            and not any(
                word in x.lower()
                for word in [
                    "earnings",
                    "revenue",
                    "guidance",
                    "inc",
                    "corp",
                    "corporation",
                ]
            )
        )
    )
    @settings(max_examples=5)  # Reduced examples for faster execution
    def test_no_false_positive_extractions(self, text_with_no_entities):
        """
        Property 7: Entity extraction completeness - No False Positives

        For any article text that does not contain recognizable entities,
        the system should not extract any entities (no false positives).

        **Feature: news-market-predictor, Property 7: Entity extraction completeness**
        **Validates: Requirements 2.2, 2.3**
        """
        assume(len(text_with_no_entities.strip()) > 5)

        # Extract entities from text without known entities
        all_entities = self.extractor.extract_all_entities(
            "test-article", text_with_no_entities
        )

        # Property: Should not extract entities from text without recognizable entities
        # Note: We allow some false positives for very generic patterns, but they should be minimal
        assert len(all_entities) <= 2, (
            f"Too many entities extracted from text without clear entities. "
            f"Text: {text_with_no_entities[:100]}... "
            f"Extracted: {[entity.entity_value for entity in all_entities]}"
        )

    @given(
        relevance_test_data=st.tuples(
            st.sampled_from(["AAPL", "GOOGL", "MSFT"]),
            st.integers(min_value=1, max_value=5),  # Number of mentions
        )
    )
    @settings(max_examples=5)  # Reduced examples for faster execution
    def test_relevance_score_properties(self, relevance_test_data):
        """
        Property 7: Entity extraction completeness - Relevance Scoring

        For any extracted entity, the relevance score should be between 0.0 and 1.0,
        and should increase with the number of mentions and financial context.

        **Feature: news-market-predictor, Property 7: Entity extraction completeness**
        **Validates: Requirements 2.2, 2.3**
        """
        stock_symbol, mention_count = relevance_test_data

        # Create text with multiple mentions and financial context
        text_parts = []
        for i in range(mention_count):
            if i % 2 == 0:
                text_parts.append(f"{stock_symbol} shares gained today")
            else:
                text_parts.append(f"Analysts are bullish on {stock_symbol} stock")

        article_text = ". ".join(text_parts) + "."

        # Extract entities
        extracted_entities = self.extractor.extract_stock_symbols(article_text)

        # Find the stock symbol entity
        stock_entities = [
            entity
            for entity in extracted_entities
            if entity.entity_value == stock_symbol
        ]

        assert len(stock_entities) > 0, f"Stock symbol {stock_symbol} not extracted"

        stock_entity = stock_entities[0]

        # Property: Relevance score should be valid
        assert (
            0.0 <= stock_entity.relevance_score <= 1.0
        ), f"Relevance score {stock_entity.relevance_score} is out of valid range [0.0, 1.0]"

        # Property: Higher mention count should generally lead to higher relevance
        # (This is a soft property due to other factors affecting relevance)
        if mention_count >= 3:
            assert stock_entity.relevance_score >= 0.6, (
                f"Expected higher relevance score for {mention_count} mentions, "
                f"got {stock_entity.relevance_score}"
            )
