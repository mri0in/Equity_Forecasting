# src/dashboard/utils.py

import logging
from typing import Optional

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
    Validate the equity string input.
    
    Args:
        equity (str): Equity ticker symbol
    
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(equity, str) and len(equity.strip()) > 0

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
