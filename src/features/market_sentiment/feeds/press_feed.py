# src/features/market_sentiment/feeds/press_feed.py

from typing import List
from datetime import datetime
import requests
import feedparser
from .base_feed import BaseFeed
from ..feed_schemas.news_item import NewsItem
from src.config.active_equity import get_active_equity
from src.utils.logger import get_logger

logger = get_logger("PressFeed")


class PressFeed(BaseFeed):
    """
    PressFeed fetches press releases and official announcements
    from multiple verified Indian and US sources for the active equity.
    """

    def __init__(self):
        super().__init__(source_name="PressFeed")

    def fetch_data(self) -> List[NewsItem]:
        ticker = get_active_equity()
        if not ticker:
            raise ValueError("Active equity ticker not set.")

        all_items: List[NewsItem] = []

        # Reliable sources (NSE, BSE, SEC, Nasdaq, NYSE, PR Newswire)
        sources = [
            # Working NSE corporate filings (India)
            (f"https://www.nseindia.com/companies-listing/corporate-filings-announcements?symbol={ticker}&tabIndex=equity", "NSE India", "html"),
            # BSE corporate announcements page (India)
            (f"https://www.bseindia.com/stock-share-price/{ticker}/corp-announcements/", "BSE India", "html"),
            # SEC filings for US tickers (requires custom headers)
            (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&dateb=&owner=exclude&count=40&output=atom", "SEC.gov", "rss"),
            # NASDAQ press releases
            (f"https://www.nasdaq.com/market-activity/stocks/{ticker}/press-releases", "Nasdaq", "html"),
            # NYSE press releases
            (f"https://www.nyse.com/quote/XNYS:{ticker}/press-releases", "NYSE", "html"),
            # PR Newswire (may not always exist per ticker)
            (f"https://www.prnewswire.com/rss/{ticker}-news.rss", "PR Newswire", "rss"),
        ]

        for url, source_name, feed_type in sources:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                if "sec.gov" in url:
                    headers = {
                        "User-Agent": "MyEquityBot/1.0 (myemail@example.com)",
                        "Accept-Encoding": "gzip, deflate",
                        "Host": "www.sec.gov"
                    }

                if feed_type in ("rss", "xml"):
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    feed = feedparser.parse(response.content)
                elif feed_type == "html":
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    feed = feedparser.parse(response.text)
                else:
                    logger.warning(f"Unknown feed type for {source_name}: {feed_type}")
                    continue

                for entry in feed.entries[:20]:
                    published = (
                        datetime(*entry.published_parsed[:6])
                        if hasattr(entry, "published_parsed")
                        else datetime.now()
                    )
                    text = (
                        entry.get("summary", "")
                        or getattr(entry, "description", "")
                        or ""
                    )

                    all_items.append(
                        NewsItem(
                            title=entry.title,
                            text=text,
                            source=source_name,
                            date=published,
                            ticker=ticker,
                            feed_name="PressFeed"
                        )
                    )

            except requests.exceptions.Timeout:
                logger.error(f"Timeout fetching press from {source_name} for {ticker}")
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP error fetching press from {source_name} for {ticker}: {e}")
            except Exception as e:
                logger.exception(f"Error parsing press feed from {source_name} for {ticker}: {e}")

        self.log_fetch(len(all_items))
        return all_items

    def validate_data(self, data: List[NewsItem]) -> bool:
        if not data:
            logger.warning("No press items fetched.")
            return False
        for item in data:
            if not all([item.title, item.text, item.source, item.date, item.ticker]):
                logger.warning(f"Invalid press item: {item}")
                return False
        return True
