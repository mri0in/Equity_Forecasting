"""
Global active equity tracker.

This module stores and retrieves the currently active equity.
Both the sentiment and forecasting modules can consume this value.
"""

from typing import Optional

# Internal variable to hold the active equity
ACTIVE_EQUITY: Optional[str] = None


def set_active_equity(ticker: str) -> None:
    """
    Set the global active equity.

    Args:
        ticker (str): Equity ticker symbol (e.g., RELIANCE, AAPL)
    """
    global ACTIVE_EQUITY
    ACTIVE_EQUITY = ticker.upper()


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
