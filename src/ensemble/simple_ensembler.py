"""
simple_ensembler.py

This module provides basic ensemble strategies such as mean, median, and weighted averaging
over multiple model predictions. It can be extended for more advanced ensemble techniques.
"""
import numpy as np
from typing import List, Dict, Callable, Literal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleEnsembler:
    """
    Class for simple ensemble strategies like mean, median, and weighted average.
    """

    def __init__(self, predictions: List[np.ndarray]):
        """
        Initialize the ensembler with predictions.

        Args:
            predictions (List[np.ndarray]): List of model prediction arrays.
        """
        if not predictions or not all(isinstance(p, np.ndarray) for p in predictions):
            raise ValueError("Predictions must be a non-empty list of numpy arrays.")
        self.predictions = predictions
        logger.info("Initialized SimpleEnsembler with %d prediction arrays.", len(predictions))

    def simple_average_ensemble(self) -> np.ndarray:
        """Perform simple mean ensemble."""
        logger.info("Performing simple average ensemble.")
        return np.mean(self.predictions, axis=0)

    def median_ensemble(self) -> np.ndarray:
        """Perform median ensemble."""
        logger.info("Performing median ensemble.")
        return np.median(self.predictions, axis=0)

    def weighted_average_ensemble(self, weights: List[float]) -> np.ndarray:
        """
        Perform weighted average ensemble.

        Args:
            weights (List[float]): Weights for each model.

        Returns:
            np.ndarray: Weighted predictions.
        """
        if len(weights) != len(self.predictions):
            raise ValueError("Number of weights must match number of prediction arrays.")
        logger.info("Performing weighted average ensemble with weights: %s", weights)
        weights = np.array(weights)
        return np.average(self.predictions, axis=0, weights=weights)

    def ensemble_predictions(
        self,
        method: Literal["mean", "median", "weighted"] = "mean",
        weights: List[float] = None,
    ) -> np.ndarray:
        """
        Select ensemble strategy.

        Args:
            method (Literal["mean", "median", "weighted"]): Ensemble method.
            weights (List[float], optional): Weights for weighted ensemble.

        Returns:
            np.ndarray: Final ensemble predictions.
        """
        logger.info("Ensembling using method: %s", method)

        if method == "mean":
            return self.simple_average_ensemble()
        elif method == "median":
            return self.median_ensemble()
        elif method == "weighted":
            if weights is None:
                raise ValueError("Weights must be provided for weighted ensemble.")
            return self.weighted_average_ensemble(weights)
        else:  # Literal makes this unreachable, but kept for safety
            raise ValueError(f"Unsupported ensemble method: {method}")

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[str],
        compute_metrics: Callable[[np.ndarray, np.ndarray, List[str]], Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Evaluate ensemble predictions using given metrics.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            metrics (List[str]): List of metrics.
            compute_metrics (Callable): Function to compute metrics.

        Returns:
            Dict[str, float]: Metric results.
        """
        logger.info("Evaluating ensemble predictions with metrics: %s", metrics)
        return compute_metrics(y_true, y_pred, metrics)
