# src/features/market_sentiment/feeds/social_feed.py
from .base_feed import BaseFeed
from typing import List, Dict
from src.utils import setup_logger

logger = setup_logger("feeds")

class SocialFeed(BaseFeed):
    """
    Fetches social media posts (e.g., Twitter/X) relevant to market sentiment.
    """

    def __init__(self, api_key: str, source_name: str = "Twitter"):
        super().__init__(source_name)
        self.api_key = api_key

    def fetch_data(self) -> List[Dict]:
        """
        Fetch social media posts.
        Returns:
            List[Dict]: Each dict represents a social post
        """
        # Placeholder: replace with actual API calls
        data = [{"text": "Stock X is soaring!", "date": "2025-09-05"}]
        self.log_fetch(len(data))
        return data

    def validate_data(self, data: List[Dict]) -> bool:
        """
        Ensure each post has required keys
        """
        valid = all("text" in item and "date" in item for item in data)
        if not valid:
            logger.warning(f"{self.source_name}: Invalid data detected")
        return valid
