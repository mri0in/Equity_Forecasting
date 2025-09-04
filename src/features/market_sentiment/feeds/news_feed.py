# src/features/market_sentiment/feeds/news_feed.py
from .base_feed import BaseFeed
from typing import List, Dict
from src.utils import setup_logger

logger = setup_logger("feeds")


class NewsFeed(BaseFeed):
    """
    Fetches news articles from a specific news API or RSS feed.
    """

    def __init__(self, api_key: str, source_name: str = "NewsAPI"):
        super().__init__(source_name)
        self.api_key = api_key

    def fetch_data(self) -> List[Dict]:
        """
        Fetch news articles.
        Returns:
            List[Dict]: Each dict represents a news item
        """
        # Placeholder: replace with actual API calls
        data = [{"title": "Sample news", "text": "Market is bullish", "date": "2025-09-05"}]
        self.log_fetch(len(data))
        return data

    def validate_data(self, data: List[Dict]) -> bool:
        """
        Basic validation: ensure required keys exist
        """
        valid = all("title" in item and "text" in item for item in data)
        if not valid:
            logger.warning(f"{self.source_name}: Invalid data detected")
        return valid
