"""
Evaluation utilities for regression and financial forecasting models.

Includes standard regression metrics as well as financial metrics like
Sharpe Ratio and Directional Accuracy.
"""

from typing import List, Dict, Union, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ----------------------------- #
# Metric Implementations
# ----------------------------- #

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return root_mean_squared_error(y_true, y_pred)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_percentage_error(y_true, y_pred)

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred)

def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    returns = y_pred - y_true
    if np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns)

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]))

# ----------------------------- #
# Metric Registry (Extensible)
# ----------------------------- #

metric_registry: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
    "r2": r2,
    "sharpe": sharpe_ratio,
    "directional_accuracy": directional_accuracy,
}

# ----------------------------- #
# Main Evaluation Entry Point
# ----------------------------- #

def compute_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    metrics: List[str] = ["rmse", "mae"]
) -> Dict[str, float]:
    """
    Compute selected evaluation metrics using dynamic registry.

    Args:
        y_true (Union[pd.Series, np.ndarray]): Ground truth values.
        y_pred (Union[pd.Series, np.ndarray]): Predicted values.
        metrics (List[str]): List of metric keys to evaluate.

    Returns:
        Dict[str, float]: Metric name to score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        logger.error(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        logger.error("Empty input arrays received.")
        raise ValueError("y_true and y_pred must not be empty.")

    results: Dict[str, float] = {}
    for name in metrics:
        metric_func = metric_registry.get(name)
        if metric_func:
            try:
                results[name] = metric_func(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Failed to compute metric {name}: {e}")
        else:
            logger.warning(f"Metric '{name}' not found in registry; skipping.")

    logger.info("Computed metrics: %s", results)
    return results
