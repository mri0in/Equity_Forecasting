import importlib
import yaml
import numpy as np
from typing import Tuple, Type

from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    """
    Encapsulates the training process of an equity forecasting model.

    Responsibilities:
    - Load config from YAML
    - Load training and optional validation data
    - Dynamically import and initialize specified model
    - Train the model (supports early stopping if enabled)
    - Save the trained model
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
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            logger.info("Loaded config from %s", path)
            return config

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and optional validation data from disk.

        Config format under 'data':
          X_train_path, y_train_path
          X_val_path, y_val_path (optional if early stopping disabled)

        Returns:
            Tuple of X_train, y_train, X_val, y_val (val may be None if not used)
        """
        data_cfg = self.config["data"]

        X_train = np.load(data_cfg["X_train_path"])
        y_train = np.load(data_cfg["y_train_path"])

        # Default to None if not using early stopping
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
        return X_train, y_train, X_val, y_val

    def _initialize_model(self):
        """
        Dynamically import and initialize model from config.

        Returns:
            Instantiated model object
        """
        model_info = self.config["model"]
        module_path = model_info["module"]
        class_name = model_info["class"]
        params = model_info.get("params", {})

        module = importlib.import_module(module_path)
        model_class: Type = getattr(module, class_name)

        logger.info(f"Instantiated model {class_name} from {module_path}")
        return model_class(params)

    def train_and_save(self):
        """
        Orchestrates full training and saving pipeline.
        """
        X, y, X_val, y_val = self.load_data()
        self.model.train(X, y, X_val, y_val)
        save_path = self.config["training"]["save_path"]
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")


def run_training(config_path: str):
    """
    CLI entrypoint to training pipeline.

    Args:
        config_path (str): Path to YAML config
    """
    trainer = ModelTrainer(config_path)
    trainer.train_and_save()
