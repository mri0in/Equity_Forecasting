# src/features/market_sentiment/feeds/press_feed.py

import csv
from typing import List
from datetime import datetime
import requests
import feedparser
from bs4 import BeautifulSoup

from .base_feed import BaseFeed
from ..feed_schemas.news_item import NewsItem
from src.config.active_equity import get_active_equity
from src.utils.logger import get_logger

logger = get_logger("PressFeed")


class PressFeed(BaseFeed):
    """
    PressFeed fetches press releases and official statements for the active equity.
    Supports multiple sources: NSE, BSE, NASDAQ, NYSE, FMP API, Finnhub API.
    """

    def __init__(self):
        super().__init__(source_name="PressFeed")

    def fetch_data(self) -> List[NewsItem]:
        ticker = get_active_equity()
        if not ticker:
            raise ValueError("Active equity ticker not set.")

        all_items: List[NewsItem] = []

        sources = [
            # NSE CSV API (new method handles session and headers)
            (f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={ticker}&csv=true",
             "NSE India", "csv_api"),
            # BSE corporate announcements page (HTML)
            (f"https://www.bseindia.com/stock-share-price/{ticker}/corp-announcements/",
             "BSE India", "html"),
            # NASDAQ press releases (HTML)
            (f"https://www.nasdaq.com/market-activity/stocks/{ticker}/press-releases",
             "Nasdaq", "html"),
            # NYSE press releases (HTML)
            (f"https://www.nyse.com/quote/XNYS:{ticker}/press-releases",
             "NYSE", "html"),
            # FinancialModelingPrep API JSON
            (f"https://financialmodelingprep.com/api/v3/press-releases/{ticker}?apikey=YOUR_FMP_API_KEY",
             "FinancialModelingPrep", "api/json"),
            # Finnhub API JSON
            (f"https://finnhub.io/api/v1/press-releases?symbol={ticker}&token=YOUR_FINNHUB_API_KEY",
             "Finnhub", "api/json"),
        ]

        for url, source_name, feed_type in sources:
            try:
                if feed_type == "html":
                    items = self.fetch_html_page(url, ticker)
                elif feed_type == "csv_api":
                    items = self.fetch_csv_api(url, ticker, source_name)
                elif feed_type == "api/json":
                    items = self.fetch_api_json(url, ticker)
                elif feed_type in ("rss", "xml"):
                    items = self.fetch_xml_feed(url, ticker)
                else:
                    logger.warning(f"Unknown feed type {feed_type} for {source_name}")
                    items = []

                all_items.extend(items)

            except Exception as e:
                logger.error(f"Error fetching press from {source_name} for {ticker}: {e}")

        self.log_fetch(len(all_items))
        return all_items

    def fetch_csv_api(self, url: str, ticker: str, source_name: str) -> List[NewsItem]:
        """
        Fetch NSE corporate announcements in CSV format and extract press releases.
        Uses session + headers to avoid 401 Unauthorized.
        """
        items: List[NewsItem] = []

        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/csv,application/vnd.ms-excel,application/json;q=0.9,*/*;q=0.8",
            "Referer": "https://www.nseindia.com",
            "Origin": "https://www.nseindia.com"
        }

        try:
            # Initial GET to set cookies
            session.get("https://www.nseindia.com", headers=headers, timeout=10)

            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            reader = csv.DictReader(response.text.splitlines())
            for row in list(reader)[:20]:  # limit to first 20 announcements
                if row.get("SUBJECT") != "Press Release":
                    continue  # only press releases

                title = row.get("SUBJECT") or ""
                text = row.get("DETAILS") or ""
                date_str = row.get("BROADCAST DATE/TIME") or ""
                try:
                    published = datetime.strptime(date_str, "%d-%b-%Y %H:%M:%S")
                except Exception:
                    published = datetime.now()

                if title and text:
                    items.append(
                        NewsItem(
                            title=title,
                            text=text,
                            source=source_name,
                            date=published,
                            ticker=ticker,
                            feed_name="PressFeed"
                        )
                    )

        except Exception as e:
            logger.error(f"Error fetching CSV press releases from {source_name} for {ticker}: {e}")

        return items

    def fetch_html_page(self, url: str, ticker: str) -> List[NewsItem]:
        """
        Fetch press items from HTML pages (BSE, NASDAQ, NYSE)
        """
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        items: List[NewsItem] = []

        for elem in soup.find_all(["a", "h3"], limit=20):
            title = elem.get_text(strip=True)
            link = elem.get("href", url)
            published = datetime.now()  # fallback if no date found
            if title:
                items.append(
                    NewsItem(
                        title=title,
                        text=title,
                        source=url.split("/")[2],
                        date=published,
                        ticker=ticker,
                        feed_name="PressFeed"
                    )
                )
        return items

    def fetch_api_json(self, url: str, ticker: str) -> List[NewsItem]:
        """
        Fetch press items from API JSON sources (FMP, Finnhub)
        """
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        items: List[NewsItem] = []

        for entry in data[:20]:
            title = entry.get("title", "")
            text = entry.get("text", "") or entry.get("summary", "")
            date_str = entry.get("date", entry.get("publishedAt")) or entry.get("datetime")
            try:
                published = datetime.fromisoformat(date_str.replace("Z", "+00:00")) if date_str else datetime.now()
            except Exception:
                published = datetime.now()

            if title and text:
                items.append(
                    NewsItem(
                        title=title,
                        text=text,
                        source=url.split("/")[2],
                        date=published,
                        ticker=ticker,
                        feed_name="PressFeed"
                    )
                )
        return items

    def fetch_xml_feed(self, url: str, ticker: str) -> List[NewsItem]:
        """
        Placeholder for XML/RSS feeds if needed later
        """
        items: List[NewsItem] = []
        feed = feedparser.parse(url)
        for entry in feed.entries[:20]:
            published = (
                datetime(*entry.published_parsed[:6])
                if hasattr(entry, "published_parsed")
                else datetime.now()
            )
            text = entry.get("summary", "") or getattr(entry, "description", "") or ""
            if getattr(entry, "title", "") and text:
                items.append(
                    NewsItem(
                        title=entry.title,
                        text=text,
                        source=url.split("/")[2],
                        date=published,
                        ticker=ticker,
                        feed_name="PressFeed"
                    )
                )
        return items

    def validate_data(self, data: List[NewsItem]) -> bool:
        if not data:
            logger.warning("No press items fetched.")
            return False
        for item in data:
            if not all([item.title, item.text, item.source, item.date, item.ticker]):
                logger.warning(f"Invalid press item: {item}")
                return False
        return True
