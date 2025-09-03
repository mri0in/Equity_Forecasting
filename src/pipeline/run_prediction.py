"""
Pipeline script for running inference using a trained model and saving predictions.

- Dynamically loads model class and checkpoint based on config.
- Runs prediction on specified dataset (validation/test).
- Saves predictions to configured path.

⚠️ IMPORTANT WARNING FOR USERS & DEVELOPERS
# For orchestration and end-user workflows, DO NOT call these classes
# directly. Instead, always use the wrapper functions in:
#
#     src/pipeline/pipeline_wrapper.py
#
# Example:
#     from src.pipeline.pipeline_wrapper import run_prediction
#     run_prediction("configs/predict_config.yaml")
#
# Reason:
# The wrappers provide a consistent interface for the orchestrator and enforce
# config-driven execution across the project. Direct class calls may bypass
# orchestration safeguards (retries, logging, markers).
# -------
"""

import os
import numpy as np
import pandas as pd
from typing import Literal

from predictor.predict import ModelPredictor
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_features(feature_path: str) -> np.ndarray:
    """
    Load features for inference from a CSV file.

    Args:
        feature_path (str): Path to feature CSV file.

    Returns:
        np.ndarray: Feature array of shape [samples, seq_len * features]
    """
    if not os.path.exists(feature_path):
        logger.error(f"Feature file not found at {feature_path}")
        raise FileNotFoundError(f"Feature file not found at {feature_path}")

    df = pd.read_csv(feature_path)
    logger.info(f"Loaded features from {feature_path} with shape {df.shape}")

    return df.values


def run_prediction_pipeline(
    config_path: str,
    paths_path: str,
    split: Literal["test", "validation"] = "test"
) -> None:
    """
    Run prediction on either test or validation data.

    Args:
        config_path (str): Path to config.yaml
        paths_path (str): Path to paths.yaml
        split (Literal["test", "validation"]): Data split to predict on
    """
    config = load_config(config_path)
    paths = load_config(paths_path)

    # Define feature file and save path
    feature_file = paths["features"].get(f"{split}_features")
    save_dir = config["predictions"].get(f"{split}_dir")
    save_path = os.path.join(save_dir, "predictions.csv")

    if feature_file is None or save_dir is None:
        raise ValueError(f"Missing path for {split} features or predictions in config.")

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load model and data
    predictor = ModelPredictor(config_path=config_path, paths_path=paths_path)
    features = load_features(feature_file)

    # Run prediction
    predictions = predictor.predict(features)

    # Save predictions
    predictor.save_predictions(predictions, save_path)
    logger.info(f"{split.capitalize()} predictions saved to: {save_path}")
