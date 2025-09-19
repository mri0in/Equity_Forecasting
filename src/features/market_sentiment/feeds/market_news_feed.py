# src/features/market_sentiment/feeds/news_feed.py

from typing import List
from datetime import datetime
import feedparser
from src.features.market_sentiment.feeds.base_feed import BaseFeed
from src.features.market_sentiment.feed_schemas.news_item import NewsItem
from src.utils import setup_logger

logger = setup_logger("MarketNewsFeed")


class MarketNewsFeed(BaseFeed):
    """
    MarketNewsFeed fetches macro-level, equity-agnostic news articles
    from both global and Indian RSS feeds.

    This feed is designed for capturing *market-wide sentiment*,
    not tied to a specific equity ticker.
    """

    def __init__(self) -> None:
        """
        Initialize MarketNewsFeed with predefined macro-level RSS sources.
        """
        super().__init__(source_name="MarketNewsFeed")
        self.sources = [
            # Global macro
            ("http://feeds.reuters.com/reuters/businessNews", "Reuters Business"),
            ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US", "Yahoo Finance Global"),
            ("https://www.marketwatch.com/rss/topstories", "MarketWatch"),

            # Indian macro
            ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "Economic Times"),
            ("https://www.business-standard.com/rss/economy-policy-10101.rss", "Business Standard"),
            ("https://www.moneycontrol.com/rss/MCtopnews.xml", "Moneycontrol"),
            ("https://www.livemint.com/rss/economy", "LiveMint"),
        ]

    def fetch_data(self) -> List[NewsItem]:
        """
        Fetch macro-level news from predefined RSS sources.

        Returns:
            List[NewsItem]: List of news items across all macro sources.
        """
        all_news: List[NewsItem] = []

        for url, source_name in self.sources:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:  # fetch max 10 per source
                    published = (
                        datetime(*entry.published_parsed[:6])
                        if hasattr(entry, "published_parsed")
                        else datetime.now()
                    )

                    # Ticker is irrelevant here, set as "MARKET"
                    news_item = NewsItem(
                        title=entry.title,
                        text=entry.get("summary", ""),
                        source=source_name,
                        date=published,
                        ticker="MARKET"
                    )
                    all_news.append(news_item)

            except Exception as e:
                logger.error(f"Error fetching news from {source_name}: {e}")

        self.log_fetch(len(all_news))
        return all_news

    def validate_data(self, data: List[NewsItem]) -> bool:
        """
        Validate fetched macro news data.

        Args:
            data (List[NewsItem]): List of fetched news items.

        Returns:
            bool: True if all items are valid, else False.
        """
        if not data:
            logger.warning("No news items fetched.")
            return False

        for item in data:
            if not all([item.title, item.text, item.source, item.date, item.ticker]):
                logger.warning(f"Invalid news item detected: {item}")
                return False

        return True
