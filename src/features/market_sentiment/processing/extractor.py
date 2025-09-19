# src/features/market_sentiment/processing/extractor.py
"""
Extractor Module (Processing Layer)

This module provides functionality to extract financially relevant
text from raw or preprocessed feed content. It applies heuristics,
rules, or NLP techniques to keep only meaningful content for
sentiment analysis.

Core Responsibilities:
    - Accept preprocessed text from feeds (news, social, press, web).
    - Filter and retain financially relevant phrases or sentences.
    - Remove non-financial, noisy, or irrelevant segments.
    - Provide extendable architecture for future ML/NLP extractors.
    - Log detailed process flow for monitoring and debugging.

Output:
    List of extracted text snippets ready for sentiment analysis.

Directory Context:
    src/features/market_sentiment/processing/
"""

import re
from typing import List

from src.utils.logger import get_logger

# Configure logger
logger = get_logger("processing_extractor")


class Extractor:
    """
    Extracts relevant financial text snippets from preprocessed content.

    Current Implementation:
        - Uses regex keyword matching for financial terms.
        - Can be extended with NLP-based extractors (NER, topic modeling, etc.).
    """

    def __init__(self, keywords: List[str] = None):
        """
        Initialize Extractor.

        Args:
            keywords (List[str], optional):
                List of financial keywords to retain relevant text.
                Defaults to a basic set if not provided.
        """
        default_keywords = [
            "revenue", "profit", "loss", "growth", "decline",
            "forecast", "guidance", "merger", "acquisition",
            "investment", "market", "earnings", "cashflow",
            "debt", "valuation", "regulation"
        ]
        self.keywords = keywords or default_keywords
        self.pattern = re.compile(r"\b(" + "|".join(self.keywords) + r")\b", re.IGNORECASE)

        logger.info("Extractor initialized with financial keywords.")

    def extract_relevant_text(self, text: str) -> str:
        """
        Extract relevant financial segments from a single text.

        Args:
            text (str): Preprocessed input text.

        Returns:
            str: Extracted text containing relevant keywords.
                 Empty string if no relevance found.
        """
        if not isinstance(text, str):
            logger.warning("Invalid input: expected string for extraction.")
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)
        relevant_sentences = [s for s in sentences if self.pattern.search(s)]

        logger.debug(
            f"Extracted {len(relevant_sentences)} relevant sentences "
            f"out of {len(sentences)} total."
        )

        return " ".join(relevant_sentences).strip()

    def extract_batch(self, texts: List[str]) -> List[str]:
        """
        Extract relevant text for a batch of documents.

        Args:
            texts (List[str]): List of preprocessed text strings.

        Returns:
            List[str]: List of extracted relevant snippets.
        """
        if not isinstance(texts, list):
            raise TypeError("Expected a list of strings for batch extraction.")

        return [self.extract_relevant_text(t) for t in texts if isinstance(t, str)]
