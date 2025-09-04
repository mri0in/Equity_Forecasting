# src/features/market_sentiment/processing/aggregator.py
from typing import List, Dict
from collections import defaultdict
from src.utils import setup_logger

logger = setup_logger("processing")

class Aggregator:
    """
    Aggregates extracted information from multiple feed items.
    Can summarize tickers, keywords, or sentiment scores.
    """

    def __init__(self):
        pass

    def aggregate_tickers(self, extracted_data: List[Dict]) -> Dict[str, int]:
        """
        Count occurrences of each ticker across multiple items.

        Args:
            extracted_data (List[Dict]): Each dict should have a 'tickers' key with a list of tickers.

        Returns:
            Dict[str, int]: Ticker -> count
        """
        ticker_counts = defaultdict(int)
        for item in extracted_data:
            for ticker in item.get("tickers", []):
                ticker_counts[ticker] += 1
        logger.info(f"Aggregated tickers: {dict(ticker_counts)}")
        return dict(ticker_counts)

    def aggregate_keywords(self, extracted_data: List[Dict]) -> Dict[str, int]:
        """
        Count occurrences of each keyword across multiple items.

        Args:
            extracted_data (List[Dict]): Each dict should have a 'keywords' key with a list of keywords.

        Returns:
            Dict[str, int]: Keyword -> count
        """
        keyword_counts = defaultdict(int)
        for item in extracted_data:
            for keyword in item.get("keywords", []):
                keyword_counts[keyword] += 1
        logger.info(f"Aggregated keywords: {dict(keyword_counts)}")
        return dict(keyword_counts)

    def aggregate_batch(self, extracted_data: List[Dict]) -> Dict[str, Dict[str, int]]:
        """
        Aggregate both tickers and keywords for a batch.

        Returns:
            Dict[str, Dict[str, int]]: {'tickers': {...}, 'keywords': {...}}
        """
        logger.info(f"Aggregating batch of {len(extracted_data)} items")
        return {
            "tickers": self.aggregate_tickers(extracted_data),
            "keywords": self.aggregate_keywords(extracted_data)
        }
