# src/features/market_sentiment/feeds/news_feed.py

from typing import List
from datetime import datetime
import feedparser
from src.config.active_equity import get_active_equity
from src.features.market_sentiment.feeds.base_feed import BaseFeed
from src.features.market_sentiment.feed_schemas.news_item import NewsItem
from src.dashboard.utils import isEquityIndOrUs
from src.utils.logger import get_logger

logger = get_logger("NewsFeed")


class NewsFeed(BaseFeed):
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
        super().__init__(source_name="NewsFeed")

    def fetch_data(self) -> List[NewsItem]:
        """
        Fetch macro-level news from predefined RSS sources.

        Returns:
            List[NewsItem]: List of news items across all macro sources.
        """
        ticker = get_active_equity()


        if not ticker:
            raise ValueError("Active equity ticker not set.")

        all_news: List[NewsItem] = []

        us_sources = [
        (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US", "Yahoo Finance"),
        (f"https://www.marketwatch.com/rss/headlines?s={ticker}", "MarketWatch"),
        (f"https://www.reuters.com/companies/{ticker}.O/rss", "Reuters"),
        (f"https://www.bloomberg.com/feeds/{ticker}.xml", "Bloomberg (if available)"),
        ]

        in_sources = [
        (f"https://economictimes.indiatimes.com/markets/stocks/news/{ticker}.cms", "Economic Times"),
        (f"https://www.business-standard.com/rss/company/{ticker}.rss", "Business Standard"),
        (f"https://www.livemint.com/rss/markets/{ticker}.xml", "LiveMint"),
        (f"https://www.moneycontrol.com/rss/stockpricefeed/{ticker}.xml", "MoneyControl"),
        (f"https://www.ndtv.com/business/stock/{ticker}/news", "NDTV Profit"),]

        if isEquityIndOrUs() == "IND":
            sources = in_sources 
        elif isEquityIndOrUs() == "USA":
            sources = us_sources    
        else:
            sources = us_sources + in_sources

        for url, source_name in sources:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:50]:  
                    published = (
                        datetime(*entry.published_parsed[:6])
                        if hasattr(entry, "published_parsed")
                        else datetime.now()
                    )
                    text = entry.get("summary", "") or getattr(entry, "description", "") or ""
                    all_news.append(
                        NewsItem(
                        title=entry.title,
                        text=text,
                        source=source_name,
                        date=published,
                        ticker=ticker,
                        feed_name="NewsFeed"
                        )
                    )
                    

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
