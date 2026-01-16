"""
Content processor implementation for cleaning and normalizing news article text.
"""

import re
import logging
from typing import Dict, Any
from ..interfaces import ContentProcessor
from ..models import NewsArticle
from ..exceptions import ContentProcessingError


logger = logging.getLogger(__name__)


class NewsContentProcessor(ContentProcessor):
    """Concrete implementation of content processing for news articles."""

    def __init__(self):
        """Initialize the content processor."""
        # Common patterns for cleaning text
        self.html_pattern = re.compile(r"<[^>]+>")
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        self.whitespace_pattern = re.compile(r"\s+")
        self.special_chars_pattern = re.compile(
            r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']+"
        )

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize article text.

        Args:
            text: Raw article text to clean

        Returns:
            Cleaned and normalized text

        Raises:
            ProcessingError: If text cleaning fails
        """
        try:
            if not isinstance(text, str):
                raise ContentProcessingError(f"Expected string input, got {type(text)}")

            if not text.strip():
                logger.warning("Empty or whitespace-only text provided for cleaning")
                return ""

            # Remove HTML tags
            cleaned = self.html_pattern.sub(" ", text)

            # Remove URLs
            cleaned = self.url_pattern.sub(" ", cleaned)

            # Remove email addresses
            cleaned = self.email_pattern.sub(" ", cleaned)

            # Remove excessive special characters but keep basic punctuation
            cleaned = self.special_chars_pattern.sub(" ", cleaned)

            # Normalize whitespace
            cleaned = self.whitespace_pattern.sub(" ", cleaned)

            # Strip leading/trailing whitespace
            cleaned = cleaned.strip()

            logger.debug(f"Cleaned text from {len(text)} to {len(cleaned)} characters")
            return cleaned

        except Exception as e:
            logger.error(f"Failed to clean text: {e}")
            raise ContentProcessingError(f"Text cleaning failed: {e}") from e

    def extract_metadata(self, article: NewsArticle) -> Dict[str, Any]:
        """
        Extract metadata from article content.

        Args:
            article: NewsArticle to extract metadata from

        Returns:
            Dictionary containing extracted metadata

        Raises:
            ProcessingError: If metadata extraction fails
        """
        try:
            if not isinstance(article, NewsArticle):
                raise ContentProcessingError(
                    f"Expected NewsArticle, got {type(article)}"
                )

            # Validate article first
            article.validate()

            metadata = {
                "word_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "has_urls": False,
                "has_emails": False,
                "content_length": 0,
                "title_length": 0,
                "cleaned_content_length": 0,
            }

            # Basic content metrics
            metadata["content_length"] = len(article.content)
            metadata["title_length"] = len(article.title)

            # Clean the content for analysis
            cleaned_content = self.clean_text(article.content)
            metadata["cleaned_content_length"] = len(cleaned_content)

            if cleaned_content:
                # Word count
                words = cleaned_content.split()
                metadata["word_count"] = len(words)

                # Sentence count (approximate)
                sentences = re.split(r"[.!?]+", cleaned_content)
                metadata["sentence_count"] = len([s for s in sentences if s.strip()])

                # Paragraph count (approximate)
                paragraphs = cleaned_content.split("\n")
                metadata["paragraph_count"] = len([p for p in paragraphs if p.strip()])

            # Check for URLs and emails in original content
            metadata["has_urls"] = bool(self.url_pattern.search(article.content))
            metadata["has_emails"] = bool(self.email_pattern.search(article.content))

            logger.debug(f"Extracted metadata for article {article.id}: {metadata}")
            return metadata

        except Exception as e:
            logger.error(
                f"Failed to extract metadata from article {getattr(article, 'id', 'unknown')}: {e}"
            )
            raise ContentProcessingError(f"Metadata extraction failed: {e}") from e

    def validate_content(self, article: NewsArticle) -> bool:
        """
        Validate that article content is processable.

        Args:
            article: NewsArticle to validate

        Returns:
            True if content is processable, False otherwise
        """
        try:
            if not isinstance(article, NewsArticle):
                logger.error(f"Expected NewsArticle, got {type(article)}")
                return False

            # First validate the article structure
            try:
                article.validate()
            except Exception as e:
                logger.error(f"Article validation failed: {e}")
                return False

            # Check if content is not empty after cleaning
            cleaned_content = self.clean_text(article.content)
            if not cleaned_content.strip():
                logger.warning(
                    f"Article {article.id} has no processable content after cleaning"
                )
                return False

            # Check minimum content length (at least 10 characters)
            if len(cleaned_content) < 10:
                logger.warning(
                    f"Article {article.id} has insufficient content length: {len(cleaned_content)}"
                )
                return False

            # Check if title is reasonable
            if not article.title.strip() or len(article.title.strip()) < 5:
                logger.warning(f"Article {article.id} has insufficient title length")
                return False

            logger.debug(f"Article {article.id} passed content validation")
            return True

        except Exception as e:
            logger.error(
                f"Content validation failed for article {getattr(article, 'id', 'unknown')}: {e}"
            )
            return False

    def process_content(self, article: NewsArticle) -> NewsArticle:
        """
        Process article content by cleaning text and extracting metadata.

        Args:
            article: NewsArticle to process

        Returns:
            Processed NewsArticle with cleaned content and metadata

        Raises:
            ContentProcessingError: If processing fails
        """
        try:
            if not isinstance(article, NewsArticle):
                raise ContentProcessingError(
                    f"Expected NewsArticle, got {type(article)}"
                )

            # Validate content first
            if not self.validate_content(article):
                logger.warning("Article %s failed content validation", article.id)
                # Return article as-is if validation fails
                return article

            # Clean the content
            cleaned_content = self.clean_text(article.content)

            # Extract metadata
            metadata = self.extract_metadata(article)

            # Create a new article with cleaned content and updated metadata
            processed_article = NewsArticle(
                id=article.id,
                title=article.title,
                content=cleaned_content,
                url=article.url,
                published_at=article.published_at,
                source=article.source,
                category=article.category,
                raw_metadata={**article.raw_metadata, "processing_metadata": metadata},
            )

            logger.debug("Successfully processed article %s", article.id)
            return processed_article

        except Exception as e:
            logger.error(
                "Failed to process content for article %s: %s",
                getattr(article, "id", "unknown"),
                e,
            )
            raise ContentProcessingError(f"Content processing failed: {e}") from e
