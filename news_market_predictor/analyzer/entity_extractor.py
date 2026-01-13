"""
Entity extraction component for identifying stock symbols, companies, and financial metrics.
"""

import re
import logging
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

from ..interfaces import EntityExtractor
from ..models import ExtractedEntity


logger = logging.getLogger(__name__)


@dataclass
class StockSymbolDatabase:
    """Simple database of known stock symbols and company mappings."""

    # Major stock symbols and their companies
    STOCK_SYMBOLS = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "GOOG": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation",
        "JPM": "JPMorgan Chase & Co.",
        "JNJ": "Johnson & Johnson",
        "V": "Visa Inc.",
        "PG": "Procter & Gamble Co.",
        "UNH": "UnitedHealth Group Inc.",
        "HD": "Home Depot Inc.",
        "MA": "Mastercard Inc.",
        "BAC": "Bank of America Corp.",
        "PFE": "Pfizer Inc.",
        "KO": "Coca-Cola Co.",
        "ABBV": "AbbVie Inc.",
        "PEP": "PepsiCo Inc.",
        "COST": "Costco Wholesale Corp.",
        "AVGO": "Broadcom Inc.",
        "WMT": "Walmart Inc.",
        "DIS": "Walt Disney Co.",
        "TMO": "Thermo Fisher Scientific Inc.",
        "ABT": "Abbott Laboratories",
        "ACN": "Accenture PLC",
        "VZ": "Verizon Communications Inc.",
        "ADBE": "Adobe Inc.",
        "NFLX": "Netflix Inc.",
        "CRM": "Salesforce Inc.",
        "CMCSA": "Comcast Corp.",
        "XOM": "Exxon Mobil Corp.",
        "NKE": "Nike Inc.",
        "CVX": "Chevron Corp.",
        "LLY": "Eli Lilly & Co.",
        "ORCL": "Oracle Corp.",
        "WFC": "Wells Fargo & Co.",
        "AMD": "Advanced Micro Devices Inc.",
        "INTC": "Intel Corp.",
        "IBM": "International Business Machines Corp.",
        "GE": "General Electric Co.",
        "F": "Ford Motor Co.",
        "GM": "General Motors Co.",
        "UBER": "Uber Technologies Inc.",
        "LYFT": "Lyft Inc.",
        "SNAP": "Snap Inc.",
        "TWTR": "Twitter Inc.",
        "SQ": "Block Inc.",
        "PYPL": "PayPal Holdings Inc.",
        "SHOP": "Shopify Inc.",
        "ZM": "Zoom Video Communications Inc.",
        "ROKU": "Roku Inc.",
        "SPOT": "Spotify Technology SA",
        "PINS": "Pinterest Inc.",
        "DOCU": "DocuSign Inc.",
        "CZR": "Caesars Entertainment Inc.",
        "MGM": "MGM Resorts International",
        "WYNN": "Wynn Resorts Ltd.",
        "LVS": "Las Vegas Sands Corp.",
        "DKNG": "DraftKings Inc.",
        "PENN": "Penn Entertainment Inc.",
    }

    # Market sectors mapping
    SECTOR_MAPPING = {
        "AAPL": "Technology",
        "GOOGL": "Technology",
        "GOOG": "Technology",
        "MSFT": "Technology",
        "AMZN": "Consumer Discretionary",
        "TSLA": "Consumer Discretionary",
        "META": "Technology",
        "NVDA": "Technology",
        "JPM": "Financial Services",
        "JNJ": "Healthcare",
        "V": "Financial Services",
        "PG": "Consumer Staples",
        "UNH": "Healthcare",
        "HD": "Consumer Discretionary",
        "MA": "Financial Services",
        "BAC": "Financial Services",
        "PFE": "Healthcare",
        "KO": "Consumer Staples",
        "ABBV": "Healthcare",
        "PEP": "Consumer Staples",
        "COST": "Consumer Staples",
        "AVGO": "Technology",
        "WMT": "Consumer Staples",
        "DIS": "Communication Services",
        "TMO": "Healthcare",
        "ABT": "Healthcare",
        "ACN": "Technology",
        "VZ": "Communication Services",
        "ADBE": "Technology",
        "NFLX": "Communication Services",
        "CRM": "Technology",
        "CMCSA": "Communication Services",
        "XOM": "Energy",
        "NKE": "Consumer Discretionary",
        "CVX": "Energy",
        "LLY": "Healthcare",
        "ORCL": "Technology",
        "WFC": "Financial Services",
        "AMD": "Technology",
        "INTC": "Technology",
        "IBM": "Technology",
        "GE": "Industrials",
        "F": "Consumer Discretionary",
        "GM": "Consumer Discretionary",
        "UBER": "Technology",
        "LYFT": "Technology",
        "SNAP": "Technology",
        "TWTR": "Technology",
        "SQ": "Technology",
        "PYPL": "Technology",
        "SHOP": "Technology",
        "ZM": "Technology",
        "ROKU": "Technology",
        "SPOT": "Technology",
        "PINS": "Technology",
        "DOCU": "Technology",
        "CZR": "Consumer Discretionary",
        "MGM": "Consumer Discretionary",
        "WYNN": "Consumer Discretionary",
        "LVS": "Consumer Discretionary",
        "DKNG": "Consumer Discretionary",
        "PENN": "Consumer Discretionary",
    }


class FinancialEntityExtractor(EntityExtractor):
    """Implementation of entity extraction for financial news articles."""

    def __init__(self):
        """Initialize the entity extractor with patterns and databases."""
        self.stock_db = StockSymbolDatabase()

        # Regex patterns for stock symbols
        self.stock_symbol_patterns = [
            # Standard format: $SYMBOL or (SYMBOL)
            r"\$([A-Z]{1,5})\b",
            r"\(([A-Z]{1,5})\)",
            # NYSE/NASDAQ format: SYMBOL:
            r"\b([A-Z]{1,5}):\s",
            # Ticker format in text
            r"\bticker\s+([A-Z]{1,5})\b",
            r"\bsymbol\s+([A-Z]{1,5})\b",
            # Common financial news format
            r"\b([A-Z]{1,5})\s+shares?\b",
            r"\b([A-Z]{1,5})\s+stock\b",
            # Trading and investment context
            r"\btrading\s+in\s+([A-Z]{1,5})\b",
            r"\binvesting\s+in\s+([A-Z]{1,5})\b",
            r"\bwatching\s+([A-Z]{1,5})\s+stock\b",
            r"\bwatching\s+([A-Z]{1,5})\s+closely\b",
            # General mention patterns
            r"\b([A-Z]{1,5})\s+was\s+active\b",
            r"\b([A-Z]{1,5})\s+gained?\b",
            r"\b([A-Z]{1,5})\s+rose\b",
            r"\b([A-Z]{1,5})\s+fell\b",
            r"\b([A-Z]{1,5})\s+dropped\b",
            # Analyst and upgrade patterns
            r"\bupgraded\s+([A-Z]{1,5})\s+to\b",
            r"\bdowngraded\s+([A-Z]{1,5})\s+to\b",
            r"\banalysts?\s+.*?([A-Z]{1,5})\s+to\b",
            # General stock symbol pattern (more permissive)
            r"\b([A-Z]{1,5})\s+(?:to\s+buy|to\s+sell|announced|reported|gained|lost)\b",
        ]

        # Patterns for financial metrics
        self.financial_metric_patterns = {
            "earnings": [
                r"earnings?\s+(?:per\s+share\s+)?(?:of\s+)?\$?([\d.,]+)",
                r"EPS\s+(?:of\s+)?\$?([\d.,]+)",
                r"quarterly\s+earnings?\s+(?:of\s+)?\$?([\d.,]+)",
                r"reported\s+earnings?\s+(?:of\s+)?\$?([\d.,]+)",
            ],
            "revenue": [
                r"revenue\s+(?:of\s+)?\$?([\d.,]+)\s*(?:billion|million|thousand)?",
                r"sales\s+(?:of\s+)?\$?([\d.,]+)\s*(?:billion|million|thousand)?",
                r"quarterly\s+revenue\s+(?:of\s+)?\$?([\d.,]+)",
                r"total\s+revenue\s+(?:of\s+)?\$?([\d.,]+)",
                r"revenue\s+reached\s+\$?([\d.,]+)\s*(?:billion|million|thousand)?",
                r"quarterly\s+revenue\s+reached\s+\$?([\d.,]+)\s*(?:billion|million|thousand)?",
            ],
            "guidance": [
                r"guidance\s+(?:of\s+)?\$?([\d.,]+)",
                r"forecast\s+(?:of\s+)?\$?([\d.,]+)",
                r"outlook\s+(?:of\s+)?\$?([\d.,]+)",
                r"expects?\s+(?:revenue\s+of\s+)?\$?([\d.,]+)",
                r"projects?\s+(?:earnings?\s+of\s+)?\$?([\d.,]+)",
            ],
        }

        # Company name patterns (simple approach)
        self.company_indicators = [
            r"\b(\w+(?:\s+\w+)*)\s+(?:Inc\.?|Corp\.?|Corporation|Company|Co\.?|Ltd\.?|LLC)\b",
            r"\b(\w+(?:\s+\w+)*)\s+(?:Group|Holdings?|Enterprises?|Industries?)\b",
        ]

    def extract_stock_symbols(self, text: str) -> List[ExtractedEntity]:
        """Extract stock symbols from article text using regex patterns."""
        entities = []
        text_upper = text.upper()

        # Track found symbols to avoid duplicates
        found_symbols = set()

        for pattern in self.stock_symbol_patterns:
            matches = re.finditer(pattern, text_upper, re.IGNORECASE)
            for match in matches:
                symbol = match.group(1).upper()

                # Skip if already found or not in our database
                if symbol in found_symbols or symbol not in self.stock_db.STOCK_SYMBOLS:
                    continue

                found_symbols.add(symbol)

                # Calculate relevance based on context
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end].strip()

                # Higher relevance if mentioned multiple times or in financial context
                relevance = self._calculate_stock_relevance(symbol, text, context)

                entities.append(
                    ExtractedEntity(
                        article_id="",  # Will be set by caller
                        entity_type="stock_symbol",
                        entity_value=symbol,
                        relevance_score=relevance,
                        context=context,
                    )
                )

        return entities

    def identify_companies(self, text: str) -> List[ExtractedEntity]:
        """Identify company names in article text using NER patterns."""
        entities = []
        found_companies = set()

        # First, check for companies associated with known stock symbols
        for symbol, company_name in self.stock_db.STOCK_SYMBOLS.items():
            # Look for exact company name matches
            company_pattern = re.escape(company_name)
            matches = re.finditer(company_pattern, text, re.IGNORECASE)

            for match in matches:
                if company_name.lower() in found_companies:
                    continue

                found_companies.add(company_name.lower())

                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end].strip()

                relevance = self._calculate_company_relevance(
                    company_name, text, context
                )

                entities.append(
                    ExtractedEntity(
                        article_id="",  # Will be set by caller
                        entity_type="company",
                        entity_value=company_name,
                        relevance_score=relevance,
                        context=context,
                    )
                )

        # Then look for other company patterns
        for pattern in self.company_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                company_name = match.group(1).strip()

                # Skip if too short or already found
                if len(company_name) < 3 or company_name.lower() in found_companies:
                    continue

                # Skip common false positives
                if company_name.lower() in [
                    "the",
                    "and",
                    "for",
                    "with",
                    "this",
                    "that",
                ]:
                    continue

                found_companies.add(company_name.lower())

                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end].strip()

                relevance = self._calculate_company_relevance(
                    company_name, text, context
                )

                entities.append(
                    ExtractedEntity(
                        article_id="",  # Will be set by caller
                        entity_type="company",
                        entity_value=company_name,
                        relevance_score=relevance,
                        context=context,
                    )
                )

        return entities

    def find_financial_metrics(self, text: str) -> List[ExtractedEntity]:
        """Find financial metrics (earnings, revenue, guidance) in text."""
        entities = []

        for metric_type, patterns in self.financial_metric_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(1)

                    # Clean up the value
                    value = value.replace(",", "")

                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end].strip()

                    # Calculate relevance based on context and value
                    relevance = self._calculate_metric_relevance(
                        metric_type, value, context
                    )

                    entities.append(
                        ExtractedEntity(
                            article_id="",  # Will be set by caller
                            entity_type="metric",
                            entity_value=f"{metric_type}: {value}",
                            relevance_score=relevance,
                            context=context,
                        )
                    )

        return entities

    def extract_all_entities(self, article_id: str, text: str) -> List[ExtractedEntity]:
        """Extract all types of entities from text and set article_id."""
        all_entities = []

        # Extract stock symbols
        stock_entities = self.extract_stock_symbols(text)
        for entity in stock_entities:
            entity.article_id = article_id
        all_entities.extend(stock_entities)

        # Extract companies
        company_entities = self.identify_companies(text)
        for entity in company_entities:
            entity.article_id = article_id
        all_entities.extend(company_entities)

        # Extract financial metrics
        metric_entities = self.find_financial_metrics(text)
        for entity in metric_entities:
            entity.article_id = article_id
        all_entities.extend(metric_entities)

        return all_entities

    def get_market_sector(self, stock_symbol: str) -> str:
        """Get market sector for a given stock symbol."""
        return self.stock_db.SECTOR_MAPPING.get(stock_symbol, "Unknown")

    def _calculate_stock_relevance(
        self, symbol: str, full_text: str, context: str
    ) -> float:
        """Calculate relevance score for a stock symbol based on context."""
        relevance = 0.5  # Base relevance

        # Count mentions in full text
        mention_count = full_text.upper().count(symbol)
        relevance += min(mention_count * 0.1, 0.3)

        # Check for financial keywords in context
        financial_keywords = [
            "earnings",
            "revenue",
            "profit",
            "loss",
            "shares",
            "stock",
            "price",
            "trading",
            "market",
            "analyst",
            "forecast",
            "guidance",
        ]

        context_lower = context.lower()
        keyword_matches = sum(
            1 for keyword in financial_keywords if keyword in context_lower
        )
        relevance += min(keyword_matches * 0.05, 0.2)

        return min(relevance, 1.0)

    def _calculate_company_relevance(
        self, company_name: str, full_text: str, context: str
    ) -> float:
        """Calculate relevance score for a company name based on context."""
        relevance = 0.4  # Base relevance for companies

        # Count mentions in full text
        mention_count = full_text.lower().count(company_name.lower())
        relevance += min(mention_count * 0.1, 0.3)

        # Check if it's a known public company
        if company_name in self.stock_db.STOCK_SYMBOLS.values():
            relevance += 0.2

        # Check for business keywords in context
        business_keywords = [
            "company",
            "corporation",
            "business",
            "firm",
            "enterprise",
            "announced",
            "reported",
            "CEO",
            "executive",
            "management",
        ]

        context_lower = context.lower()
        keyword_matches = sum(
            1 for keyword in business_keywords if keyword in context_lower
        )
        relevance += min(keyword_matches * 0.05, 0.1)

        return min(relevance, 1.0)

    def _calculate_metric_relevance(
        self, metric_type: str, value: str, context: str
    ) -> float:
        """Calculate relevance score for a financial metric based on context."""
        relevance = 0.6  # Base relevance for financial metrics

        # Check if value looks like a reasonable financial number
        try:
            numeric_value = float(value.replace(",", ""))
            if numeric_value > 0:
                relevance += 0.1
            if numeric_value > 1000000:  # Large numbers are more likely to be financial
                relevance += 0.1
        except ValueError:
            relevance -= 0.2  # Penalize non-numeric values

        # Check for supporting keywords in context
        supporting_keywords = {
            "earnings": ["quarterly", "annual", "per share", "EPS", "profit"],
            "revenue": ["quarterly", "annual", "sales", "income", "total"],
            "guidance": ["forecast", "outlook", "expects", "projects", "estimates"],
        }

        context_lower = context.lower()
        if metric_type in supporting_keywords:
            keyword_matches = sum(
                1
                for keyword in supporting_keywords[metric_type]
                if keyword in context_lower
            )
            relevance += min(keyword_matches * 0.05, 0.2)

        return min(relevance, 1.0)
