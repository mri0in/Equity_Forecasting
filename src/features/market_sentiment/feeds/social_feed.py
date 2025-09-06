# src/features/market_sentiment/feeds/social_feed.py

from typing import List
from datetime import datetime
from .base_feed import BaseFeed
from ..feed_schemas.news_item import NewsItem
from src.features.market_sentiment.active_equity.active_equity import ActiveEquity
from src.utils import setup_logger

logger = setup_logger("SocialFeed")


class SocialFeed(BaseFeed):
    """
    SocialFeed fetches social media posts (e.g., Twitter, Reddit) related to the active equity.
    Placeholder: implement actual API integration later.
    """

    def __init__(self):
        super().__init__("SocialFeed")
        self.active_equity = ActiveEquity()

    def fetch_data(self) -> List[NewsItem]:
        ticker = self.active_equity.get_ticker()
        if not ticker:
            raise ValueError("Active equity ticker not set.")

        all_posts: List[NewsItem] = []

        # Example placeholder: simulate fetching social posts
        dummy_posts = [
            {"title": f"{ticker} discussion on Twitter", "text": "Stock is trending...", "source": "Twitter"},
            {"title": f"{ticker} Reddit chatter", "text": "Investors talking...", "source": "Reddit"}
        ]
        for entry in dummy_posts:
            all_posts.append(
                NewsItem(
                    title=entry["title"],
                    text=entry["text"],
                    source=entry["source"],
                    date=datetime.utcnow(),
                    ticker=ticker
                )
            )

        self.log_fetch(len(all_posts))
        return all_posts

    def validate_data(self, data: List[NewsItem]) -> bool:
        if not data:
            logger.warning("No social posts fetched.")
            return False
        for item in data:
            if not all([item.title, item.text, item.source, item.date, item.ticker]):
                logger.warning(f"Invalid social item: {item}")
                return False
        return True
