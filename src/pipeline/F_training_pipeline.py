# src/pipeline/run_training.py
"""
Pipeline module to orchestrate end-to-end model training.

This script performs:
1. Config loading from model_utils
2. Dynamic model instantiation
3. Training data loadingl
4. Model training
5. Saving trained model

⚠️ IMPORTANT WARNING:
Do NOT call these classes directly in end-user workflows.
Use wrappers in src/pipeline/pipeline_wrapper.py to enforce
orchestration, logging, retries, and task markers.
"""

import numpy as np
from typing import Tuple
from src.utils.logger import get_logger
from src.utils.model_utils import load_config_and_model
from src.monitoring.monitor import TrainingMonitor

logger = get_logger(__name__)
monitor = TrainingMonitor()


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
        monitor.log_stage_start("init_pipeline", {"config_path": config_path})
        self.config, self.model = load_config_and_model(config_path)
        monitor.log_stage_end("init_pipeline")

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training features and targets from .npy files.

        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features), y (targets)
        """
        monitor.log_stage_start("load_training_data")
        try:
            x_path = self.config["data"]["X_train_path"]
            y_path = self.config["data"]["y_train_path"]
            X = np.load(x_path)
            y = np.load(y_path)
            logger.info("Loaded training data: X=%s, y=%s", X.shape, y.shape)
        except Exception as e:
            logger.exception("Error loading training data")
            monitor.log_stage_end("load_training_data", success=False)
            raise
        monitor.log_stage_end("load_training_data")
        return X, y

    def run(self) -> None:
        """
        Run the training pipeline:
        - Load data
        - Train model
        - Save model
        """
        monitor.log_stage_start("run_training_pipeline")
        try:
            X, y = self.load_training_data()
            self.model.train(X, y)

            save_path = self.config["training"]["save_path"]
            self.model.save_model(save_path)
            logger.info("Model training completed and saved to: %s", save_path)
        except Exception as e:
            logger.exception("Training pipeline failed")
            monitor.log_stage_end("run_training_pipeline", success=False)
            raise
        monitor.log_stage_end("run_training_pipeline")


def run_training_pipeline(config_path: str) -> None:
    """
    Function-based entry point for external importers.

    Args:
        config_path (str): Path to YAML configuration
    """
    monitor.log_stage_start("entry_run_training_pipeline", {"config_path": config_path})
    pipeline = ModelTrainerPipeline(config_path)
    pipeline.run()
    monitor.log_stage_end("entry_run_training_pipeline")
