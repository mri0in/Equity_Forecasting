# src/features/market_sentiment/feeds/news_feed.py

from typing import List
from datetime import datetime
import feedparser
from src.features.market_sentiment.feeds.base_feed import BaseFeed
from src.features.market_sentiment.feed_schemas.news_item import NewsItem
from src.features.market_sentiment.active_equity.active_equity import ActiveEquity
from src.utils import setup_logger

logger = setup_logger("NewsFeed")


class NewsFeed(BaseFeed):
    """
    NewsFeed fetches equity-related news articles from multiple RSS sources
    (Google News and Yahoo Finance) for the currently active equity.
    """

    def __init__(self):
        """
        Initialize NewsFeed. The ticker is dynamically obtained from ActiveEquity.
        """
        super().__init__(source_name="NewsFeed")
        self.active_equity = ActiveEquity()

    def fetch_data(self) -> List[NewsItem]:
        """
        Fetch news articles for the currently active ticker.

        Returns:
            List[NewsItem]: List of news items for the active equity.

        Raises:
            ValueError: If no active ticker is set.
        """
        ticker = self.active_equity.get_ticker()
        if not ticker:
            raise ValueError("Active equity ticker not set. Cannot fetch news.")

        all_news: List[NewsItem] = []

        # Define sources
        sources = [
            (f"https://news.google.com/rss/search?q={ticker}+stock", "Google News"),
            (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN", "Yahoo Finance")
        ]

        # Fetch news from all sources
        for url, source_name in sources:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:  # limit per source
                    published = (
                        datetime(*entry.published_parsed[:6])
                        if hasattr(entry, "published_parsed")
                        else datetime.utcnow()
                    )

                    news_item = NewsItem(
                        title=entry.title,
                        text=entry.get("summary", ""),
                        source=source_name,
                        date=published,
                        ticker=ticker
                    )
                    all_news.append(news_item)

            except Exception as e:
                logger.error(f"Error fetching news from {source_name} for {ticker}: {e}")

        self.log_fetch(len(all_news))
        return all_news

    def validate_data(self, data: List[NewsItem]) -> bool:
        """
        Validate fetched news data.

        Args:
            data (List[NewsItem]): List of news items.

        Returns:
            bool: True if all items have required fields, else False.
        """
        if not data:
            logger.warning("No news items fetched.")
            return False

        for item in data:
            if not all([item.title, item.text, item.source, item.date, item.ticker]):
                logger.warning(f"Invalid news item detected: {item}")
                return False
        return True
