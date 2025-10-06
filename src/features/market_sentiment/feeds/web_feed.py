# src/features/market_sentiment/feeds/web_feed.py

from typing import List
from datetime import datetime
import feedparser
from .base_feed import BaseFeed
from ..feed_schemas.news_item import NewsItem
from src.config.active_equity import get_active_equity
from src.utils.logger import get_logger

logger = get_logger("WebFeed")


class WebFeed(BaseFeed):
    """
    WebFeed fetches general web-based news for the active equity from RSS sources.
    """

    def __init__(self):
        super().__init__("WebFeed")

    def fetch_data(self) -> List[NewsItem]:
        ticker = get_active_equity()
        if not ticker:
            raise ValueError("Active equity ticker not set.")

        all_items: List[NewsItem] = []

        sources = [
        (f"https://www.moneycontrol.com/rss/company/{ticker}/news.xml", "MoneyControl"),
        (f"https://www.financialexpress.com/market/{ticker}-rss/", "Financial Express"),
        (f"https://news.google.com/rss/search?q={ticker}+stock+news", "Google News"),
        (f"https://inshorts.com/en/read/{ticker}", "InShorts"),
        ]

        for url, source_name in sources:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:
                    published = (
                        datetime(*entry.published_parsed[:6])
                        if hasattr(entry, "published_parsed") else datetime.utcnow()
                    )
                    all_items.append(
                        NewsItem(
                            title=entry.title,
                            text=entry.get("summary", ""),
                            source=source_name,
                            date=published,
                            ticker=ticker,
                            feed_name="WebFeed"
                        )
                    )
            except Exception as e:
                logger.error(f"Error fetching web news from {source_name} for {ticker}: {e}")

        self.log_fetch(len(all_items))
        return all_items

    def validate_data(self, data: List[NewsItem]) -> bool:
        if not data:
            logger.warning("No web news fetched.")
            return False
        for item in data:
            if not all([item.title, item.text, item.source, item.date, item.ticker]):
                logger.warning(f"Invalid web item: {item}")
                return False
        return True
