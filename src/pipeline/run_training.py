# src/pipeline/run_training.py

"""
Pipeline module to orchestrate end-to-end model training.

This script performs:
1. Config loading from model_util of utils folder
2. Dynamic model instantiation
3. Training data loading
4. Model training
5. Saving trained model

"""

import numpy as np
from typing import Tuple
from src.utils.logger import get_logger
from src.utils.model_utils import load_config_and_model

logger = get_logger(__name__)

class ModelTrainerPipeline:
    """
    Pipeline class for training any model defined via configuration.
    """

    def __init__(self, config_path: str):
        """
        Constructor: loads configuration and model instance.

        Args:
            config_path (str): Path to YAML config file
        """
        self.config, self.model = load_config_and_model(config_path)

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training features and targets from .npy files.

        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features), y (targets)
        """
        try:
            x_path = self.config["data"]["X_train_path"]
            y_path = self.config["data"]["y_train_path"]
            X = np.load(x_path)
            y = np.load(y_path)
            logger.info("Loaded training data: X=%s, y=%s", X.shape, y.shape)
            return X, y
        except Exception as e:
            logger.exception("Error loading training data")
            raise

    def run(self) -> None:
        """
        Run the training pipeline:
        - Load data
        - Train model
        - Save model
        """
        try:
            X, y = self.load_training_data()
            self.model.train(X, y)

            save_path = self.config["training"]["save_path"]
            self.model.save_model(save_path)
            logger.info("Model training completed and saved to: %s", save_path)

        except Exception as e:
            logger.exception("Training pipeline failed")
            raise


def run_training_pipeline(config_path: str) -> None:
    """
    Function-based entry point for external importers.

    Args:
        config_path (str): Path to YAML configuration
    """
    pipeline = ModelTrainerPipeline(config_path)
    pipeline.run()
