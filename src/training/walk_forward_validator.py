"""
Walk-Forward Validation Module

This module performs expanding window walk-forward validation for time series forecasting.
It trains the given model multiple times on increasing subsets of data and evaluates 
predictions using multiple financial metrics.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from src.models.base_model import BaseModel
from src.training.evaluate import compute_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)

class WalkForwardValidator:
    """
    Performs walk-forward validation using either expanding or rolling windows.
    """

    def __init__(
        self,
        model: BaseModel,
        window_type: str = "expanding",  # Options: "expanding", "rolling"
        window_size: int = 100,
        step_size: int = 1,
        metrics: List[str] = ["rmse", "mae"],
        early_stopping: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the WalkForwardValidator.

        Args:
            model (BaseModel): Instance of a model implementing BaseModel.
            window_type (str): "expanding" or "rolling" window type.
            window_size (int): Initial window size for training.
            step_size (int): Step to slide the window forward.
            metrics (List[str]): List of metric keys to evaluate.
            early_stopping (Optional[Dict]): Dict with 'patience' and 'delta' keys.
        """
        self.model = model
        self.window_type = window_type
        self.window_size = window_size
        self.step_size = step_size
        self.metrics = metrics
        self.early_stopping = early_stopping

        if self.window_type not in ["expanding", "rolling"]:
            logger.error("Invalid window_type: %s", self.window_type)
            raise ValueError("window_type must be 'expanding' or 'rolling'")

    def validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Run walk-forward validation over the dataset.

        Args:
            X (np.ndarray or pd.DataFrame): Features.
            y (np.ndarray or pd.Series): Target values.

        Returns:
            Dict[str, float]: Averaged metric results across all folds.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = len(X)
        fold_metrics: List[Dict[str, float]] = []

        logger.info("Starting walk-forward validation with %s window", self.window_type)

        for start in range(self.window_size, n_samples - 1, self.step_size):
            if self.window_type == "expanding":
                train_X = X[:start]
                train_y = y[:start]
            else:  # rolling
                train_X = X[start - self.window_size:start]
                train_y = y[start - self.window_size:start]

            test_X = X[start:start + self.step_size]
            test_y = y[start:start + self.step_size]

            if len(test_X) == 0 or len(test_y) == 0:
                logger.warning("Skipping fold due to empty test set.")
                continue

            try:
                # Early stopping parameters passed to model
                self.model.train(
                    train_X,
                    train_y,
                    val_X=test_X,
                    val_y=test_y,
                    early_stopping=self.early_stopping
                )
                preds = self.model.predict(test_X)
                fold_result = compute_metrics(test_y, preds, self.metrics)
                fold_metrics.append(fold_result)
                logger.info("Fold %d evaluated: %s", start, fold_result)
            except Exception as e:
                logger.error("Error during fold %d: %s", start, str(e))

        return self._average_metrics(fold_metrics)

    def _average_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Compute average of each metric over all folds.

        Args:
            all_metrics (List[Dict[str, float]]): Per-fold metric results.

        Returns:
            Dict[str, float]: Averaged metrics.
        """
        if not all_metrics:
            logger.warning("No valid folds found during walk-forward validation.")
            return {}

        avg_result: Dict[str, float] = {}
        for key in self.metrics:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_result[key] = np.mean(values)

        logger.info("Final averaged WFV metrics: %s", avg_result)
        return avg_result
