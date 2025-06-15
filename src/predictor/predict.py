import os
import numpy as np
import pandas as pd
from typing import Union
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
import importlib

logger = get_logger(__name__)


class ModelPredictor:
    """
    Predictor class for running inference using a saved model.
    Supports dynamic model loading based on config.
    """

    def __init__(
        self,
        config_path: str,
        paths_path: str,
        model_checkpoint: str = None,
    ) -> None:
        """
        Load model for inference.

        Args:
            config_path (str): Path to config.yaml
            paths_path (str): Path to paths.yaml
            model_checkpoint (str, optional): Explicit checkpoint path. Defaults to None.
        """
        self.config = load_config(config_path)
        self.paths = load_config(paths_path)

        if model_checkpoint is not None:
            checkpoint_path = model_checkpoint
        else:
            checkpoint_path = self.config.get("early_stopping", {}).get("checkpoint_path")
            if checkpoint_path is None:
                raise ValueError("No model checkpoint path provided or found in config under 'early_stopping.checkpoint_path'.")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        # Dynamically load the model class
        module_path = self.config["model"]["module"]
        class_name = self.config["model"]["class"]
        model_class = self._dynamic_import(module_path, class_name)

        logger.info(f"Loading model checkpoint from {checkpoint_path}")
        self.model = model_class.load_model(checkpoint_path)
        logger.info("Model successfully loaded for inference.")

    def _dynamic_import(self, module_path: str, class_name: str):
        """
        Dynamically import a class from a module.

        Args:
            module_path (str): Module path (e.g. 'src.models.lstm_model')
            class_name (str): Class name (e.g. 'LSTMModel')

        Returns:
            class: The class object
        """
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run predictions on input data.

        Args:
            X (np.ndarray): Input features (e.g., shape: [samples, seq_len, features])

        Returns:
            np.ndarray: Predicted values
        """
        logger.info("Running prediction on input of shape %s", X.shape)
        predictions = self.model.predict(X)
        logger.info("Prediction completed.")
        return predictions

    def save_predictions(self, predictions: np.ndarray, save_path: str) -> None:
        """
        Save predictions to disk as a CSV file.

        Args:
            predictions (np.ndarray): Predicted values array.
            save_path (str): Full path to save CSV file.
        """
        try:
            if predictions.ndim == 1:
                df = pd.DataFrame(predictions, columns=["prediction"])
            else:
                num_cols = predictions.shape[1]
                df = pd.DataFrame(predictions, columns=[f"pred_{i}" for i in range(num_cols)])

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"Predictions saved successfully to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save predictions to {save_path}: {e}")
            raise

    def save_predictions_from_type(self, predictions: np.ndarray, split_type: str) -> None:
        """
        Save predictions based on split type using default save locations from paths.yaml.

        Args:
            predictions (np.ndarray): Predicted values.
            split_type (str): 'test' or 'validation'
        """
        dir_key = f"{split_type}_dir"
        try:
            pred_dir = self.paths["predictions"][dir_key]
        except KeyError:
            raise ValueError(f"Invalid split_type '{split_type}'. Expecting 'test' or 'validation'.")

        os.makedirs(pred_dir, exist_ok=True)
        save_path = os.path.join(pred_dir, "predictions.csv")
        self.save_predictions(predictions, save_path)
