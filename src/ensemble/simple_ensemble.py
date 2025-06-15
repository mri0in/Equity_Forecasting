"""
simple_ensemble.py

This module provides basic ensemble strategies such as mean, median, and weighted averaging
over multiple model predictions. It can be extended for more advanced ensemble techniques.
"""
import numpy as np
from typing import List, Literal, Union
from src.utils.logger import get_logger
from src.models.evaluate import evaluate_metrics

logger = get_logger(__name__)

class SimpleEnsembler:
    """
    Performs simple ensembling methods like mean, median, or weighted average.
    """

    def __init__(
        self,
        method: Literal["mean", "median", "weighted"] = "mean",
        weights: Union[List[float], None] = None,
        metrics: List[str] = ["rmse", "mae"]
    ):
        """
        Initialize the SimpleEnsembler.

        Args:
            method (str): Ensemble method - "mean", "median", or "weighted".
            weights (List[float], optional): Weights for "weighted" method.
            metrics (List[str]): Evaluation metrics to compute.
        """
        self.method = method
        self.weights = weights
        self.metrics = metrics

    def ensemble_predictions(self, prediction_arrays: List[np.ndarray]) -> np.ndarray:
        """
        Combine predictions using the selected ensemble method.

        Args:
            prediction_arrays (List[np.ndarray]): List of model prediction arrays.

        Returns:
            np.ndarray: Combined predictions.
        """
        stacked = np.stack(prediction_arrays, axis=0)
        logger.info(f"Stacked predictions shape: {stacked.shape}")

        if self.method == "mean":
            return np.mean(stacked, axis=0)
        elif self.method == "median":
            return np.median(stacked, axis=0)
        elif self.method == "weighted":
            if not self.weights or len(self.weights) != len(prediction_arrays):
                raise ValueError("Invalid or missing weights for weighted ensemble.")
            weights = np.array(self.weights).reshape(-1, 1)
            return np.sum(stacked * weights, axis=0) / np.sum(weights)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.method}")

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> dict:
        """
        Evaluate the ensembled predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Ensemble predictions.

        Returns:
            dict: Dictionary of computed metrics.
        """
        return evaluate_metrics(y_true, y_pred, self.metrics)
