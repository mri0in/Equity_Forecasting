# src/features/market_sentiment/active_equity/active_equity.py

from typing import Optional
from src.utils import setup_logger

logger = setup_logger("active_equity")


class ActiveEquity:
    """
    Singleton class to manage the currently selected equity ticker.
    Ensures a single source of truth across all modules.
    """

    _instance = None
    _active_ticker: Optional[str] = None

    def __new__(cls):
        """
        Implement singleton pattern to ensure one instance only.
        """
        if cls._instance is None:
            cls._instance = super(ActiveEquity, cls).__new__(cls)
        return cls._instance

    def set_ticker(self, ticker: str) -> None:
        """
        Set the active equity ticker.

        Args:
            ticker (str): Stock symbol (e.g., "TCS.NS", "AAPL")

        Raises:
            ValueError: If ticker is empty or not a string.
        """
        if not ticker or not isinstance(ticker, str):
            logger.error(f"Invalid ticker value: {ticker}")
            raise ValueError("Ticker must be a non-empty string.")

        self._active_ticker = ticker.upper()
        logger.info(f"Active equity set to: {self._active_ticker}")

    def get_ticker(self) -> Optional[str]:
        """
        Get the current active equity ticker.

        Returns:
            Optional[str]: Current ticker or None if not set.
        """
        if self._active_ticker is None:
            logger.warning("Active equity ticker has not been set yet.")
        return self._active_ticker

    def clear_ticker(self) -> None:
        """
        Clear the currently active ticker.
        """
        logger.info(f"Clearing active equity: {self._active_ticker}")
        self._active_ticker = None
