# src/features/market_sentiment/active_equity/active_equity.py

from typing import Optional
from src.utils import setup_logger
from src.config import active_equity as config_equity  # Single source of truth

logger = setup_logger("active_equity")


class ActiveEquity:
    """
    Singleton class to manage the currently selected equity ticker.
    Reads from the config file as the authoritative source of truth.
    Backward-compatible get_ticker for feeds and other modules.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ActiveEquity, cls).__new__(cls)
        return cls._instance

    def set_ticker(self, ticker: str) -> None:
        """
        Set the active equity ticker.
        Updates the config file value to maintain single source of truth.

        Args:
            ticker (str): Stock symbol (e.g., "TCS.NS", "AAPL")
        """
        if not ticker or not isinstance(ticker, str):
            logger.error(f"Invalid ticker value: {ticker}")
            raise ValueError("Ticker must be a non-empty string.")

        config_equity.ACTIVE_TICKER = ticker.upper()
        logger.info(f"Active equity updated in config: {config_equity.ACTIVE_TICKER}")

    def get_ticker(self) -> Optional[str]:
        """
        Get the current active equity ticker from the config file.

        Returns:
            Optional[str]: Current ticker or None if not set.
        """
        ticker = getattr(config_equity, "ACTIVE_TICKER", None)
        if not ticker:
            logger.warning("Active equity ticker not set in config.")
        return ticker

    def clear_ticker(self) -> None:
        """
        Clear the active ticker in the config file.
        """
        logger.info(f"Clearing active equity: {getattr(config_equity, 'ACTIVE_TICKER', None)}")
        config_equity.ACTIVE_TICKER = None
