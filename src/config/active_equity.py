# src/features/market_sentiment/active_equity/active_equity.py

from typing import Optional
import yfinance as yf
from src.dashboard.utils import validate_equity
from src.utils.logger import get_logger

logger = get_logger("active_equity")

# ==========================================================
# Global state
# ==========================================================
ACTIVE_EQUITY: Optional[str] = None

class ActiveEquity:
    """
    Singleton class to manage the currently selected equity ticker.
    Reads from the config file as the authoritative source of truth.
    Now supports validation and boolean-return flow for UI integration.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ActiveEquity, cls).__new__(cls)
        return cls._instance

def set_active_equity(ticker: str) -> bool:
    """
    Validate and set the active equity symbol.

    Args:
        ticker (str): The equity ticker symbol.

    Returns:
        bool: True if successfully validated and set, False otherwise.
    """
    global ACTIVE_EQUITY

    try:
        if not ticker or not isinstance(ticker, str):
            logger.warning("No valid ticker provided for set_active_equity.")
            return False

        # Run validation (wrapped in try/except internally)
        if validate_equity(ticker):
            ACTIVE_EQUITY = ticker.upper()
            logger.info(f"Active equity set to: {ACTIVE_EQUITY}")
            return True
        else:
            logger.warning(f"Invalid ticker attempted: {ticker}")
            return False

    except Exception as e:
        logger.error(f"Unexpected error while setting active equity: {e}", exc_info=True)
        return False

def get_active_equity() -> str:
    """
    Retrieve the global active equity.

    Returns:
        str: Currently active equity ticker.

    Raises:
        ValueError: If no equity has been set.
    """
    if ACTIVE_EQUITY is None:
        raise ValueError("Active equity has not been set.")
    return ACTIVE_EQUITY

def clear_active_equity() -> None:
    """
    Clear the currently set active equity.
    """
    global ACTIVE_EQUITY
    ACTIVE_EQUITY = None
    logger.info("Active equity cleared.")
