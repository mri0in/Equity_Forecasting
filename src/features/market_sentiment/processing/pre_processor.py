# src/features/market_sentiment/processing/pre_processor.py
import re
import nltk
from typing import List
from nltk.corpus import stopwords
from src.utils.logger import get_logger

logger = get_logger("processing")

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


"""
Pre-processing module for raw market sentiment text data.

This module is responsible for cleaning and normalizing text
before it is passed into the NLP pipeline. It removes noise,
standardizes text, and ensures input is ready for downstream
sentiment analysis.
Core Responsibilities:
    - Remove HTML tags, URLs, special characters, and punctuation.
    - Normalize casing (lowercase).
    - Remove stopwords (using NLTK).
    - Tokenize text into words.
    - Provide batch processing for lists of texts.
    - Log detailed process flow for monitoring and debugging.
"""

import re
import logging
from typing import List
from nltk.corpus import stopwords

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TextPreProcessor:
    """
    Preprocess raw text for market sentiment analysis.

    Responsibilities:
    - Normalize casing (lowercase).
    - Remove URLs, HTML tags, special characters, and numbers.
    - Remove stopwords (using NLTK).
    - Output clean tokenized text ready for NLP.

    Example:
        processor = TextPreProcessor()
        cleaned = processor.clean_text("TCS reports RECORD profits! Visit http://example.com")
        # "tcs reports record profits"
    """

    def __init__(self, language: str = "english") -> None:
        """
        Initialize the preprocessor with stopwords.
        
        Args:
            language (str): Language for stopword filtering. Default = "english".
        """
        self.stop_words = set()  # default empty set to avoid crashes
        try:
            self.stop_words = set(stopwords.words(language))
            logger.debug("Stopwords loaded for language: %s", language)
        except LookupError:
            logger.warning("Stopwords not found, attempting to download...")
            try:
                nltk.download("stopwords", quiet=True)
                self.stop_words = set(stopwords.words(language))
                logger.info("Stopwords successfully downloaded for language: %s", language)
            except Exception as e:
                logger.error("Failed to download stopwords: %s", e)
                logger.warning("Continuing without stopwords; text processing may be affected.")

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize a single text string.

        Args:
            text (str): Raw text input.

        Returns:
            str: Cleaned text string.
        """
        if not isinstance(text, str):
            logger.error("Input must be of type str, got %s", type(text))
            raise ValueError("Input must be a string.")

        logger.debug("Original text: %s", text)

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove punctuation and special characters
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Tokenize and remove stopwords
        tokens = [word for word in text.split() if word not in self.stop_words]

        cleaned_text = " ".join(tokens)
        logger.info("Cleaned Text: %s", cleaned_text)
        logger.debug("Cleaned text: %s", cleaned_text)

        return cleaned_text

    def batch_clean(self, texts: List[str]) -> List[str]:
        """
        Clean and normalize a batch of text strings.

        Args:
            texts (List[str]): List of raw text inputs.

        Returns:
            List[str]: List of cleaned text strings.
        """
        if not isinstance(texts, list):
            logger.error("Input must be of type list, got %s", type(texts))
            raise ValueError("Input must be a list of strings.")

        cleaned_batch = []
        for t in texts:
            try:
                cleaned_batch.append(self.clean_text(t))
            except Exception as e:
                logger.warning("Skipping text due to error: %s", str(e))
                continue

        return cleaned_batch
