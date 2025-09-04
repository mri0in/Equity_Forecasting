# src/features/market_sentiment/processing/extractor.py
from typing import List, Dict
import re
from src.utils import setup_logger

logger = setup_logger("processing")

class Extractor:
    """
    Extracts entities, keywords, and stock tickers from text.
    """

    def __init__(self, tickers_list: List[str] = None):
        """
        Args:
            tickers_list (List[str], optional): List of valid stock tickers to detect.
        """
        self.tickers_list = tickers_list or []

    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from the text.

        Args:
            text (str): Preprocessed text.

        Returns:
            List[str]: Detected tickers.
        """
        if not isinstance(text, str):
            logger.warning(f"Expected string but got {type(text)}")
            return []

        # Simple pattern: uppercase words matching tickers list
        tickers_found = [word for word in text.split() if word.upper() in self.tickers_list]
        logger.info(f"Tickers found: {tickers_found}")
        return tickers_found

    def extract_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """
        Extract relevant keywords from text.

        Args:
            text (str): Preprocessed text.
            keywords (List[str]): Keywords to search for.

        Returns:
            List[str]: Detected keywords in text.
        """
        if not text or not keywords:
            return []

        found_keywords = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)]
        logger.info(f"Keywords found: {found_keywords}")
        return found_keywords

    def extract_batch(self, texts: List[str], keywords: List[str] = None) -> List[Dict]:
        """
        Extract tickers and keywords for a batch of texts.

        Args:
            texts (List[str]): List of preprocessed texts.
            keywords (List[str], optional): Keywords to search for.

        Returns:
            List[Dict]: Each dict has 'tickers' and 'keywords' keys.
        """
        logger.info(f"Extracting entities from batch of {len(texts)} texts")
        result = []
        for text in texts:
            item = {
                "tickers": self.extract_tickers(text),
                "keywords": self.extract_keywords(text, keywords or [])
            }
            result.append(item)
        return result
