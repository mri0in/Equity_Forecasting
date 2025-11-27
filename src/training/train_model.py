# src/models/train_model.py
"""
train_model.py

Handles the training pipeline for equity forecasting models.
Integrates monitoring and logging for each stage.

Responsibilities:
- Load YAML config
- Load training/validation data
- Dynamically import and initialize model
- Train the model with optional early stopping
- Save the trained model
- Report all stages to TrainingMonitor
"""

import importlib
import yaml
import numpy as np
from typing import Tuple, Type, Optional

from src.utils.logger import get_logger
from src.monitoring.monitor import TrainingMonitor

logger = get_logger(__name__)
monitor = TrainingMonitor()


class ModelTrainer:
    """
    Encapsulates the training process of an equity forecasting model.
    """

    def __init__(self, config_path: str):
        """
        Initialize ModelTrainer by loading config and initializing model.

        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config = self.load_config(config_path)
        self.model = self._initialize_model()

    def load_config(self, path: str) -> dict:
        """
        Load YAML configuration.

        Args:
            path (str): Path to config

        Returns:
            dict: Parsed config
        """
        monitor.log_stage_start("load_config", {"config_path": path})
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            logger.info("Loaded config from %s", path)
        monitor.log_stage_end("load_config")
        return config

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load training and optional validation data from disk.

        Returns:
            Tuple of X_train, y_train, X_val, y_val (val may be None if not used)
        """
        monitor.log_stage_start("load_data")
        data_cfg = self.config["data"]

        X_train = np.load(data_cfg["X_train_path"])
        y_train = np.load(data_cfg["y_train_path"])

        X_val = y_val = None
        early_stopping = self.config.get("model", {}).get("params", {}).get("early_stopping", {}).get("enabled", False)

        if early_stopping:
            try:
                X_val = np.load(data_cfg["X_val_path"])
                y_val = np.load(data_cfg["y_val_path"])
                logger.info(f"Loaded validation data X_val:{X_val.shape}, y_val:{y_val.shape}")
            except KeyError as e:
                logger.error("Validation data paths missing in config but required for early stopping.")
                raise e

        logger.info(f"Loaded training data X_train:{X_train.shape}, y_train:{y_train.shape}")
        monitor.log_stage_end("load_data")
        return X_train, y_train, X_val, y_val

    def _initialize_model(self):
        """
        Dynamically import and initialize model from config.

        Returns:
            Instantiated model object
        """
        monitor.log_stage_start("initialize_model")
        model_info = self.config["model"]
        module_path = model_info["module"]
        class_name = model_info["class"]
        params = model_info.get("params", {})

        module = importlib.import_module(module_path)
        model_class: Type = getattr(module, class_name)

        logger.info(f"Instantiated model {class_name} from {module_path}")
        monitor.log_stage_end("initialize_model")
        return model_class(params)

    def train_and_save(self):
        """
        Orchestrates full training and saving pipeline.
        """
        monitor.log_stage_start("train_and_save")
        X, y, X_val, y_val = self.load_data()
        self.model.train(X, y, X_val, y_val)
        save_path = self.config["training"]["save_path"]
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        monitor.log_stage_end("train_and_save")


def run_training(config_path: str):
    """
    CLI entrypoint to training pipeline.

    Args:
        config_path (str): Path to YAML config
    """
    monitor.log_stage_start("run_training", {"config_path": config_path})
    trainer = ModelTrainer(config_path)
    trainer.train_and_save()
    monitor.log_stage_end("run_training")
