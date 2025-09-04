# src/features/market_sentiment/feeds/base_feed.py
import abc
from typing import List, Any

from src.utils import setup_logger

logger = setup_logger("feeds")


class BaseFeed(abc.ABC):
    """
    Abstract base class for all market sentiment feeds.
    Defines the common interface for fetching and validating data.
    """

    def __init__(self, source_name: str):
        """
        Args:
            source_name (str): Name of the data source
        """
        self.source_name = source_name

    @abc.abstractmethod
    def fetch_data(self) -> List[Any]:
        """
        Fetch data from the source.
        Returns:
            List[Any]: Raw data items
        """
        pass

    @abc.abstractmethod
    def validate_data(self, data: List[Any]) -> bool:
        """
        Validate the fetched data.
        Args:
            data (List[Any]): Raw data items
        Returns:
            bool: True if data is valid, else False
        """
        pass

    def log_fetch(self, count: int):
        """
        Log the number of items fetched.
        Args:
            count (int)
        """
        logger.info(f"{self.source_name}: Fetched {count} items")
