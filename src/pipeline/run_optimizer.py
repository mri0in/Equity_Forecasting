# src/pipeline/run_optimizer.py

"""
Module to orchestrate hyperparameter optimization for equity forecasting models.

This module loads configuration, training data, selects the optimizer backend dynamically,
runs the hyperparameter search, and logs the progress.

It is designed to be called from a single CLI entry point (e.g., src/main.py).
"""

from typing import Dict, Tuple
import numpy as np
import yaml
from src.utils.logger import get_logger
from src.training.optimizers import get_optimizer  # Your factory to get optimizer by name
from src.utils.config import load_yaml_config     # Assuming this exists for config loading

logger = get_logger(__name__)


def load_training_data(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training features and targets from given file paths.

    Args:
        x_path (str): Path to training feature .npy file.
        y_path (str): Path to training target .npy file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Loaded features and targets arrays.
    """
    try:
        X = np.load(x_path)
        y = np.load(y_path)
        logger.info(f"Loaded training data: X shape {X.shape}, y shape {y.shape}")
        return X, y
    except Exception as e:
        logger.exception(f"Failed to load training data: {e}")
        raise


def run_hyperparameter_optimization(config_path: str, optimizer_name: str = "optuna") -> None:
    """
    Main orchestration function for running hyperparameter optimization.

    Args:
        config_path (str): Path to the YAML configuration file.
        optimizer_name (str, optional): Name of the optimizer backend (e.g., 'optuna', 'raytune', 'hyperopt').
                                        Defaults to 'optuna'.
    """
    # Load config using your utility
    config: Dict = load_yaml_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")

    # Load training data paths from config
    x_path = config["data"]["X_train_path"]
    y_path = config["data"]["y_train_path"]

    # Load data
    X_train, y_train = load_training_data(x_path, y_path)

    # Get the optimizer function/class dynamically
    optimizer_func = get_optimizer(optimizer_name)
    logger.info(f"Selected optimizer: {optimizer_name}")

    # Number of trials from config, fallback to default 50
    n_trials = config.get("training", {}).get("n_trials", 50)

    # Run optimization
    logger.info(f"Starting hyperparameter optimization for {n_trials} trials...")
    study = optimizer_func(config, X_train, y_train, n_trials=n_trials)
    logger.info("Hyperparameter optimization completed.")

