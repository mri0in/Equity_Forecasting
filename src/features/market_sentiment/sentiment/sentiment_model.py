#src/features/market_sentiment/sentiment/sentiment_model.py
"""
Sentiment Model Module

This module defines the SentimentModel class responsible for analyzing
text from multiple market feeds and assigning sentiment scores.

Features:
    - Supports multiple backends: TextBlob(local,fallback), Hugging Face models (FinBERT, etc.).
    - Produces discrete labels (positive, neutral, negative).
    - Produces continuous score in range [-1, 1].
    - Designed for modular extension (easy model switching).

Directory Context:
    src/features/market_sentiment/sentiment/
"""

import logging
from typing import List, Dict, Union

from textblob import TextBlob  

from src.utils.logger import get_logger
from src.features.market_sentiment.nlp_models.hf_model_manager import HFModelManager

# Configure logger
logger = get_logger("sentiment_model")


class SentimentModel:
    """
    Sentiment analysis model for financial texts.

    Backends:
        - "textblob"  → lightweight, quick polarity scoring, as fallback for 
                        testing, offline use.
        - "finbert"   → Hugging Face finance-specific transformer.
        - "hf:<name>" → Any Hugging Face model registered in ModelRegistry.
    """

    def __init__(self, model_name: str = "finbert"):
        """
        Initialize the sentiment model.

        Args:
            model_name (str): Name of the sentiment model backend.
                              Options:
                                - "textblob"
                                - "finbert"
                                - "hf:<registry_key>"
        """
        self.model_name = model_name
        self.hf_manager = HFModelManager()
        self.pipeline = None
        self.fallback_model = "textblob"

        try:
            if self.model_name == "textblob":
                logger.info("SentimentModel initialized with TextBlob backend.")

            elif self.model_name == "finbert":
                logger.info("Loading FinBERT sentiment model from Hugging Face...")
                self.pipeline = self.hf_manager.load_pipeline("finbert")

            elif self.model_name.startswith("hf:"):
                hf_key = self.model_name.split("hf:")[-1]
                logger.info(f"Loading Hugging Face model from registry: {hf_key}")
                self.pipeline = self.hf_manager.load_pipeline(hf_key)

            else:
                raise ValueError(f"Unsupported model backend: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}. Falling back to {self.fallback_model}.")
            self.model_name = self.fallback_model
            self.pipeline = None

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

            else:  # Hugging Face models (FinBERT or registry models)
                result = self.pipeline(text)[0]
                raw_label = result["label"].lower()  # e.g., "positive"

                # Convert HF label → unified schema
                label = (
                    "positive" if "pos" in raw_label else
                    "negative" if "neg" in raw_label else
                    "neutral"
                )

                # Map raw probability score into [-1, 1]
                score = float(result["score"])
                if label == "negative":
                    score = -score
                elif label == "neutral":
                    score = 0.0

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
