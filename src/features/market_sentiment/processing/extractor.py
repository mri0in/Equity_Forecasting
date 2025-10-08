# src/features/market_sentiment/processing/extractor.py
"""
Extractor Module (Processing Layer)

Enhanced Financial Text Extractor
---------------------------------
This module extracts financially relevant phrases or sentences from
preprocessed text using hybrid heuristics:
    1. Financial keyword matching (revenue, profit, etc.)
    2. Ticker/company detection (AAPL, TSLA, $INFY)
    3. Contextual heuristics for financial actions or events

    Output:
    List of extracted text snippets ready for sentiment analysis.
    
Design Principles:
    - Extendable for ML/NLP models (NER, transformers)
    - Fully logged for debugging and transparency
    - Batch-friendly for integration in the sentiment pipeline
"""

import re
from typing import List
from src.utils.logger import get_logger

# Configure logger
logger = get_logger("processing_extractor")


class Extractor:
    """
    Hybrid Extractor for Financial Text.
    Combines rule-based heuristics with pattern detection to identify
    meaningful sentences for sentiment analysis.
    """

    def __init__(self, keywords: List[str] = None):
        """
        Initialize Extractor with default financial keywords and regex patterns.

        Args:
            keywords (List[str], optional): Custom financial keyword list.
        """
        default_keywords = [
            "revenue", "profit", "loss", "growth", "decline", "forecast",
            "guidance", "merger", "acquisition", "investment", "market",
            "earnings", "cashflow", "debt", "valuation", "regulation",
            "stock", "share", "dividend", "buyback", "IPO", "capital"
        ]
        self.keywords = keywords or default_keywords
        self.keyword_pattern = re.compile(
            r"\b(" + "|".join(self.keywords) + r")\b", re.IGNORECASE
        )

        # Recognize stock tickers and symbols (AAPL, TSLA, $INFY, NSE, etc.)
        self.ticker_pattern = re.compile(
            r"(\$?[A-Z]{2,6})(?=\b|\s|,|\.|!|\?)"
        )

        # General company reference pattern (optional future use)
        self.company_pattern = re.compile(
            r"\b(Inc\.?|Ltd\.?|Corp\.?|Company|Enterprises|Holdings)\b", re.IGNORECASE
        )

        logger.info("Extractor initialized with hybrid heuristic patterns.")

    
    def extract_relevant_text(self, text: str) -> str:
        """
        Extract financially relevant sentences or tokens.

        Args:
            text (str): Input preprocessed text.

        Returns:
            str: Extracted relevant text. Empty if nothing relevant found.
        """
        if not isinstance(text, str):
            logger.warning("Invalid input: expected string for extraction.")
            return ""

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            logger.debug("Empty input text after normalization.")
            return ""

        # Split into sentences for finer granularity
        sentences = re.split(r"(?<=[.!?])\s+", text)
        relevant_sentences = []

        for sentence in sentences:
            has_keyword = bool(self.keyword_pattern.search(sentence))
            has_ticker = bool(self.ticker_pattern.search(sentence))
            has_company_ref = bool(self.company_pattern.search(sentence))

            # Keep sentence if it contains financial cues or ticker symbol
            if has_keyword or has_ticker or has_company_ref:
                relevant_sentences.append(sentence)

        # Fallback: if single-token ticker like “AAPL”
        if len(sentences) == 1 and self.ticker_pattern.fullmatch(sentences[0].strip()):
            logger.debug(f"Single-token ticker detected: {text}")
            return text.strip()

        extracted_text = " ".join(relevant_sentences).strip()

        logger.debug(
            f"Extracted {len(relevant_sentences)} relevant sentence(s) "
            f"from total {len(sentences)}."
        )

        return extracted_text


    def extract_batch(self, texts: List[str]) -> List[str]:
        """
        Apply extraction to a batch of text documents.

        Args:
            texts (List[str]): List of preprocessed strings.

        Returns:
            List[str]: List of extracted financial text segments.
        """
        if not isinstance(texts, list):
            raise TypeError("Expected a list of strings for batch extraction.")

        logger.info(f"Starting batch extraction for {len(texts)} document(s).")

        results = []
        for t in texts:
            if isinstance(t, str):
                extracted = self.extract_relevant_text(t)
                if extracted:
                    results.append(extracted)
            else:
                logger.warning("Skipped non-string entry during batch extraction.")

        logger.info(f"Batch extraction complete. Valid results: {len(results)}")
        return results

