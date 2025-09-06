# src/features/market_sentiment/feed_schemas/news_item.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from src.utils import setup_logger

logger = setup_logger("feed_schemas")

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
    """
    title: str
    text: str
    date: datetime
    source: str
    ticker: str = field(default="")  # Added ticker field

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
            logger.warning(f"NewsItem missing required fields: {missing} | {self}")
        logger.info(f"NewsItem created: {self.title} from {self.source} for {self.ticker}")