import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger.

    Args:
        name (str): Logger name, usually __name__.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Avoid adding multiple handlers
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
