#src/features/market_sentiment/sentiment/sentiment_model.py
"""
Sentiment Model Module

This module defines the SentimentModel class responsible for analyzing
text from multiple market feeds and assigning sentiment scores.

Features:
- Supports multiple backends: TextBlob, FinBERT (extendable to custom models).
- Produces discrete labels (positive, neutral, negative).
- Produces continuous score in range [-1, 1].
- Designed for modular extension (switch models easily).

Directory Context:
    src/features/market_sentiment/sentiment/
"""

import logging
from typing import List, Dict, Union

from textblob import TextBlob  # lightweight sentiment analyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.utils import setup_logger

# Configure logger
logger = setup_logger("sentiment_model")


class SentimentModel:
    """
    Sentiment analysis model for financial texts.

    Backends:
        - "textblob" → lightweight, quick polarity scoring.
        - "finbert"  → domain-specific transformer for finance sentiment.
    """

    def __init__(self, model_name: str = "textblob"):
        """
        Initialize the sentiment model.

        Args:
            model_name (str): Name of the sentiment model backend.
                              Options: ["textblob", "finbert"]
        """
        self.model_name = model_name

        if self.model_name == "finbert":
            try:
                logger.info("Loading FinBERT sentiment model...")
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "ProsusAI/finbert"
                )
                self.nlp_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer
                )
                logger.info("FinBERT successfully loaded.")
            except Exception as e:
                logger.exception("Failed to load FinBERT. Falling back to TextBlob.")
                self.model_name = "textblob"
        else:
            logger.info("SentimentModel initialized with TextBlob backend.")

    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a given text.

        Args:
            text (str): Input text to analyze.

        Returns:
            Dict[str, Union[str, float]]:
                - "label": str ("positive", "neutral", "negative")
                - "score": float (continuous sentiment score [-1, 1])
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text input received for sentiment analysis.")
            return {"label": "neutral", "score": 0.0}

        try:
            if self.model_name == "textblob":
                analysis = TextBlob(text)
                score = float(analysis.sentiment.polarity)
                label = self._map_score_to_label(score)

            elif self.model_name == "finbert":
                result = self.nlp_pipeline(text)[0]
                raw_label = result["label"].lower()  # e.g., "positive"
                score = float(result["score"]) if raw_label != "neutral" else 0.0

                # Convert FinBERT labels to our schema
                label = (
                    "positive" if raw_label == "positive"
                    else "negative" if raw_label == "negative"
                    else "neutral"
                )

                # Map score to range [-1, 1]
                score = score if label == "positive" else -score if label == "negative" else 0.0

            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            logger.info(f"Sentiment analysis completed. Label={label}, Score={score:.3f}")
            return {"label": label, "score": score}

        except Exception as e:
            logger.exception(f"Error analyzing text sentiment: {e}")
            return {"label": "neutral", "score": 0.0}

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment for a batch of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[Dict[str, Union[str, float]]]: List of sentiment results.
        """
        results = []
        for t in texts:
            results.append(self.analyze_text(t))
        return results

    @staticmethod
    def _map_score_to_label(score: float) -> str:
        """
        Map continuous polarity score to discrete sentiment label.

        Args:
            score (float): Polarity score between -1 and 1.

        Returns:
            str: Sentiment label ("positive", "neutral", "negative")
        """
        if score > 0.05:
            return "positive"
        elif score < -0.05:
            return "negative"
        else:
            return "neutral"
