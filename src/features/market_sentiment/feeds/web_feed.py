# src/features/market_sentiment/feeds/web_feed.py

from typing import List
from datetime import datetime
import feedparser
from .base_feed import BaseFeed
from ..feed_schemas.news_item import NewsItem
from src.features.market_sentiment.active_equity.active_equity import ActiveEquity
from src.utils import setup_logger

logger = setup_logger("WebFeed")


class WebFeed(BaseFeed):
    """
    WebFeed fetches general web-based news for the active equity from RSS sources.
    """

    def __init__(self):
        super().__init__("WebFeed")
        self.active_equity = ActiveEquity()

    def fetch_data(self) -> List[NewsItem]:
        ticker = self.active_equity.get_ticker()
        if not ticker:
            raise ValueError("Active equity ticker not set.")

        all_items: List[NewsItem] = []

        sources = [
            (f"https://www.moneycontrol.com/rss/company/{ticker}/press-releases.xml", "MoneyControl"),
            (f"https://www.financialexpress.com/market/{ticker}-rss/", "Financial Express"),
            (f"https://www.livemint.com/rss/{ticker}.xml", "LiveMint")
            (f"https://news.google.com/rss/search?q={ticker}+stock", "Google News")
        ]

        for url, source_name in sources:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
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
                            ticker=ticker
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
