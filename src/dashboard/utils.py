# src/dashboard/utils.py

import logging
from typing import Optional
import yfinance as yf

# -------------------------------
# Logger Function
# -------------------------------
def get_ui_logger(name: Optional[str] = "dashboard") -> logging.Logger:
    """
    Returns a configured logger for the dashboard module.
    
    Args:
        name (str): Logger name. Defaults to 'dashboard'.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False
    return logger

# -------------------------------
# Helper Functions
# -------------------------------
def validate_equity(equity: str) -> bool:
    """
    Validate if the provided equity ticker is a real, listed stock 
    (US or India) using yfinance.

    Args:
        equity (str): Equity ticker symbol (e.g., 'AAPL', 'INFY.NS')

    Returns:
        bool: True if the ticker is valid and has historical data, False otherwise.
    """
    if not isinstance(equity, str) or not equity.strip():
        return False

    logger = logging.getLogger(__name__)
    symbol = equity.strip().upper()

    # Candidate tickers in priority: US, NSE, BSE
    candidates = [symbol]

    # Add NSE and BSE variants if not already suffixed
    if not symbol.endswith(".NS"):
        candidates.append(f"{symbol}.NS")
    if not symbol.endswith(".BO"):
        candidates.append(f"{symbol}.BO")

    for candidate in candidates:
        try:
            ticker = yf.Ticker(candidate)
            hist = ticker.history(period="1d")
            if hist is not None and not hist.empty:
                logger.info(f"Equity validation passed for: {candidate}")
                return True
            else:
                logger.warning(f"No historical data found for: {candidate}")
        except Exception as e:
            logger.error(f"Error validating equity {candidate}: {e}")

    logger.warning(f"Equity validation failed for all candidates: {candidates}")
    return False

def normalize_score(score: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """
    Clip or normalize a sentiment score to be within bounds.
    
    Args:
        score (float): Input sentiment score
        min_val (float): Minimum allowed value
        max_val (float): Maximum allowed value
    
    Returns:
        float: Clipped score
    """
    return max(min(score, max_val), min_val)

# -------------------------------
# Dashboard Constants
# -------------------------------
DEFAULT_BULLET_HEIGHT = 300
DEFAULT_GAUGE_HEIGHT = 350
FEED_COLORS = {"positive": "green", "negative": "red", "neutral": "gray"}
