# src/features/market_sentiment/processing/pre_processor.py
import re
from typing import List
from src.utils import setup_logger

logger = setup_logger("processing")

# ---------------------------
# Text Cleaning Steps:
# 1. Remove all characters except letters, numbers, and spaces
#    (punctuation and special symbols are removed)
#    Regex: r"[^a-zA-Z0-9\s]" → matches everything NOT a-z, A-Z, 0-9, or whitespace
#
# 2. Normalize whitespace: replace multiple spaces/tabs/newlines with a single space
#    Then remove leading/trailing spaces using .strip()
#    Regex: r"\s+" → matches one or more whitespace characters
#    .strip() removes leading/trailing whitespace
#
# Syntax reminder: re.sub(pattern, replacement, string)
#    pattern     → regex pattern to search for
#    replacement → string to replace matches with
#    string      → input text
# 
# Example:text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#         text = re.sub(r"\s+", " ", text).strip()
#
# ---------------------------



class PreProcessor:
    """
    Preprocesses text data for sentiment analysis or feature extraction.

    Methods:
        clean_text: Cleans a single text string.
        preprocess_batch: Cleans a list of text strings.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Lowercase, remove punctuation and extra spaces.

        Args:
            text (str): Raw text input.

        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str):
            logger.warning(f"Expected string but got {type(text)}. Returning empty string.")
            return ""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Cleans a batch of text strings.

        Args:
            texts (List[str]): List of raw text inputs.

        Returns:
            List[str]: List of cleaned text strings.
        """
        logger.info(f"Preprocessing batch of {len(texts)} texts")
        return [self.clean_text(t) for t in texts]
