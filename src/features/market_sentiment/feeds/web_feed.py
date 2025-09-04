# src/features/market_sentiment/feeds/web_feed.py
from .base_feed import BaseFeed
from typing import List, Dict
from src.utils import setup_logger

logger = setup_logger("feeds")


class WebFeed(BaseFeed):
    """
    Fetches articles from websites (RSS feeds or HTML scraping).
    Can be extended to support any web-based content.
    """

    def __init__(self, feed_url: str, source_name: str = "WebFeed"):
        super().__init__(source_name)
        self.feed_url = feed_url

    def fetch_data(self) -> List[Dict]:
        """
        Fetch latest articles from the website.
        Returns:
            List[Dict]: Each dict has keys like 'title', 'text', 'date'
        """
        # Placeholder: replace with RSS parser or HTML scraper
        data = [
            {"title": "Market opens higher", "text": "Stocks rally on positive data", "date": "2025-09-05"}
        ]
        self.log_fetch(len(data))
        return data

    def validate_data(self, data: List[Dict]) -> bool:
        """
        Basic validation for required keys.
        """
        valid = all("title" in item and "text" in item and "date" in item for item in data)
        if not valid:
            logger.warning(f"{self.source_name}: Invalid data detected")
        return valid
