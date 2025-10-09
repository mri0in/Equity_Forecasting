# src/features/market_sentiment/feed_schemas/news_item.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("feed_schemas")

@dataclass
class NewsItem:
    """
    Standardized schema for a feed item (news article, social post, press release, or web article).

    Attributes:
        title (str): Headline or title of the item.
        text (str): Full content/text of the item.
        date (datetime): Publication date.
        source (str): Source name (e.g., news, social, press, web).
        ticker (Optional[str]): Related stock ticker, if available.
        url (Optional[str]): Link to the original content.
        feed_name (Optional[str]): Name of the feed that provided this item.
    """
    title: str
    text: str
    date: datetime
    source: str
    ticker: str = field(default="")  
    feed_name: Optional[str] = field(default=None)

    def __post_init__(self):
        """
        Post-initialization validation and logging.
        Ensures required fields are not empty and logs creation.
        """
        missing = []
        for attr in ["title", "text", "date", "source", "ticker"]:
            if not getattr(self, attr):
                missing.append(attr)
        if missing:
            logger.warning(f"Missing required fields for {self.feed_name} item: {missing} | {self}")
        logger.info(f"{self.feed_name} item added: '{self.title}' from {self.source} for {self.ticker}")