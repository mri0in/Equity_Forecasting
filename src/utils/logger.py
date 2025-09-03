import logging
import sys
import os
from typing import Optional

def get_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Create or retrieve a logger with optional file logging.

    Args:
        name (str): Name of the logger.
        level (str, optional): Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_file (str, optional): Path to log file. If None, logs only to console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():  # Avoid adding duplicate handlers
        # Set level from config
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure directory exists
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
