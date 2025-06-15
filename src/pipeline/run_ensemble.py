"""
This module handles the orchestration of ensemble modeling.
It loads base model predictions, applies an ensemble strategy,
evaluates the final ensemble prediction, and logs the results.

Assumptions:
- Each base model has already generated its predictions and saved them to disk.
- A simple ensemble strategy (e.g., mean or weighted average) is applied.
- Evaluation metrics follow the unified interface in `evaluate.py`.
"""
# src/pipeline/run_ensemble.py

import numpy as np
import yaml
from typing import Dict, List
from src.utils.logger import get_logger
from src.ensemble.simple_ensemble import simple_average_ensemble, weighted_average_ensemble
from src.ensemble.generate_oof import generate_oof_predictions
from src.ensemble.meta_features import create_meta_features
from src.ensemble.train_meta_features import train_meta_model
from src.ensemble.evaluate_meta_model import evaluate_meta_model
from src.training.evaluate import compute_metrics

logger = get_logger(__name__)


def load_ensemble_config(config_path: str) -> Dict:
    """
    Load ensemble configuration from a YAML file.

    Args:
        config_path (str): Path to config YAML.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded ensemble config from %s", config_path)
    return config


def run_ensemble(config: Dict) -> Dict[str, float]:
    """
    Run the selected ensemble strategy based on config.

    Args:
        config (Dict): Ensemble configuration dictionary.

    Returns:
        Dict[str, float]: Evaluation metric results.
    """
    ensemble_type = config.get("ensemble", {}).get("method", "mean").lower()
    logger.info("Running ensemble method: %s", ensemble_type)

    if ensemble_type == "mean":
        y_preds = [np.load(path) for path in config["ensemble"]["pred_paths"]]
        averaged_preds = simple_average_ensemble(y_preds)
        y_true = np.load(config["ensemble"]["y_true_path"])
        metrics = config["ensemble"].get("metrics", ["rmse", "mae"])
        results = compute_metrics(y_true, averaged_preds, metrics)
        logger.info("Mean ensemble evaluation: %s", results)
        return results

    elif ensemble_type == "weighted":
        y_preds = [np.load(path) for path in config["ensemble"]["pred_paths"]]
        weights: List[float] = config["ensemble"]["weights"]
        weighted_preds = weighted_average_ensemble(y_preds, weights)
        y_true = np.load(config["ensemble"]["y_true_path"])
        metrics = config["ensemble"].get("metrics", ["rmse", "mae"])
        results = compute_metrics(y_true, weighted_preds, metrics)
        logger.info("Weighted ensemble evaluation: %s", results)
        return results

    elif ensemble_type == "stacked":
        logger.info("Generating OOF predictions...")
        oof_preds, holdout_preds, y_holdout = generate_oof_predictions(config)

        logger.info("Creating meta features...")
        X_meta_train, X_meta_test = create_meta_features(oof_preds, holdout_preds)

        logger.info("Training meta model...")
        y_meta_pred, _ = train_meta_model(X_meta_train, y_holdout, config)

        logger.info("Evaluating meta model...")
        metrics = config["ensemble"].get("metrics", ["rmse", "mae", "sharpe", "directional"])
        results = evaluate_meta_model(y_holdout, y_meta_pred, metrics)
        return results

    else:
        logger.error("Unsupported ensemble method: %s", ensemble_type)
        raise ValueError(f"Unsupported ensemble method: {ensemble_type}")
