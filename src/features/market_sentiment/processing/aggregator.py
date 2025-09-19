# src/features/market_sentiment/processing/aggregator.py
"""
Aggregator Module (Processing Layer)

This module provides functionality to aggregate preprocessed and
feature-extracted data into a structured format ready for downstream
sentiment modeling or ML pipelines.

Core Responsibilities:
    - Accept processed items (from multiple feeds or processors).
    - Validate and normalize the structure of inputs.
    - Aggregate features into a single container (per equity).
    - Provide modular, extendable architecture for new feature types.
    - Log detailed process flow for debugging and monitoring.

Output:
    Dictionary containing aggregated features, e.g.:

    {
        "equity": "TCS",
        "aggregated_texts": [...],
        "features": {...}
    }

Directory Context:
    src/features/market_sentiment/processing/
"""

import statistics
from typing import List, Dict, Any, Optional, Union

from src.utils.logger import get_logger

# Configure logger
logger = get_logger("processing_aggregator")


class Aggregator:
    """
    Aggregates preprocessed and extracted features for a given equity.
    Designed to integrate multiple feeds and processors into a single
    normalized representation.

    Example Use:
        aggregator = Aggregator("INFY")
        result = aggregator.aggregate_texts(["good growth", "weak demand"])
    """

    def __init__(self, equity: str):
        """
        Initialize the Aggregator for a specific equity.

        Args:
            equity (str): Equity ticker or name.
        """
        if not isinstance(equity, str) or not equity.strip():
            raise ValueError("Equity name must be a non-empty string.")

        self.equity = equity
        self.aggregated_texts: List[str] = []
        self.features: Dict[str, Any] = {}

        logger.info(f"Aggregator initialized for equity: {self.equity}")

    def aggregate_texts(self, texts: List[str]) -> Dict[str, Any]:
        """
        Aggregate cleaned and extracted texts.

        Args:
            texts (List[str]): List of preprocessed/extracted texts.

        Returns:
            Dict[str, Any]: Aggregated result containing equity,
                            aggregated_texts, and features.
        """
        if not isinstance(texts, list):
            raise TypeError("Expected a list of texts for aggregation.")

        valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]

        if not valid_texts:
            logger.warning(f"No valid texts provided for equity {self.equity}.")
        else:
            logger.info(
                f"Aggregating {len(valid_texts)} valid texts for {self.equity}."
            )

        # Store for downstream
        self.aggregated_texts.extend(valid_texts)

        # Example feature: average text length
        lengths = [len(t.split()) for t in valid_texts] if valid_texts else []
        self.features["avg_text_length"] = (
            statistics.mean(lengths) if lengths else 0
        )
        self.features["num_texts"] = len(valid_texts)

        logger.debug(
            f"Equity {self.equity}: "
            f"avg_text_length={self.features['avg_text_length']}, "
            f"num_texts={self.features['num_texts']}"
        )

        return {
            "equity": self.equity,
            "aggregated_texts": self.aggregated_texts,
            "features": self.features,
        }

    def reset(self) -> None:
        """
        Reset aggregator state for reuse.
        """
        logger.info(f"Resetting aggregator for equity {self.equity}.")
        self.aggregated_texts = []
        self.features = {}
