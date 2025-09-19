# src/features/market_sentiment/feeds/social_feed.py

from typing import List
from datetime import datetime
import os
import requests
from .base_feed import BaseFeed
from ..feed_schemas.news_item import NewsItem
from src.features.market_sentiment.active_equity.active_equity import ActiveEquity
from src.utils import setup_logger

logger = setup_logger("SocialFeed")


class SocialFeed(BaseFeed):
    """
    SocialFeed fetches social media posts (Twitter/Reddit) related to the active equity.
    Priority:
        1. Twitter API (if credentials available)
        2. Reddit API (pushshift, no credentials)
        3. Dummy fallback
    """

    def __init__(self):
        super().__init__("SocialFeed")
        self.active_equity = ActiveEquity()
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")  # optional

    def _fetch_twitter(self, ticker: str) -> List[NewsItem]:
        """Fetch tweets using Twitter API v2 if credentials exist."""
        if not self.twitter_bearer_token:
            logger.info("No Twitter credentials, skipping Twitter API.")
            return []

        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}
        params = {"query": ticker, "max_results": 5, "tweet.fields": "created_at"}

        resp = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code != 200:
            logger.warning(f"Twitter API error {resp.status_code}: {resp.text}")
            return []

        tweets = []
        for tw in resp.json().get("data", []):
            tweets.append(
                NewsItem(
                    title=f"Tweet on {ticker}",
                    text=tw.get("text", ""),
                    source="Twitter",
                    date=datetime.fromisoformat(tw["created_at"].replace("Z", "+00:00")),
                    ticker=ticker,
                )
            )
        return tweets[:10]

    def _fetch_reddit(self, ticker: str) -> List[NewsItem]:
        """Fetch Reddit posts via pushshift (no creds)."""
        url = f"https://api.pushshift.io/reddit/search/comment/?q={ticker}&size=3"
        resp = requests.get(url, timeout=5)

        if resp.status_code != 200 or "data" not in resp.json():
            logger.warning("Reddit API returned no data.")
            return []

        posts = []
        for entry in resp.json()["data"]:
            posts.append(
                NewsItem(
                    title=f"{ticker} Reddit Post",
                    text=entry.get("body", ""),
                    source="Reddit",
                    date=datetime.now(),
                    ticker=ticker,
                )
            )
        return posts[:10]

    def _fallback_dummy(self, ticker: str) -> List[NewsItem]:
        """Return dummy social posts if no real feed works."""
        logger.info("Falling back to dummy social posts.")
        dummy_posts = [
            {"title": f"{ticker} discussion on Twitter", "text": "Stock is trending...", "source": "Twitter"},
            {"title": f"{ticker} Reddit chatter", "text": "Investors talking...", "source": "Reddit"},
        ]
        return [
            NewsItem(
                title=entry["title"],
                text=entry["text"],
                source=entry["source"],
                date=datetime.now(),
                ticker=ticker,
            )
            for entry in dummy_posts
        ]

    def fetch_data(self) -> List[NewsItem]:
        ticker = self.active_equity.get_ticker()
        if not ticker:
            raise ValueError("Active equity ticker not set.")

        all_posts: List[NewsItem] = []

        # Try Twitter & Reddit posts
        twitter_posts = self._fetch_twitter(ticker)
        reddit_posts = self._fetch_reddit(ticker)

        # Combine both if available
        all_posts.extend(twitter_posts)
        all_posts.extend(reddit_posts)

        # If both are empty → fallback
        if not all_posts:
            all_posts = self._fallback_dummy(ticker)     
        

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
