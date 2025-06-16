from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    Defines a unified interface and shared model parameter handling.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize model with parameters and set up logging.

        Args:
            model_params (Optional[Dict[str, Any]]): Configuration dictionary for model and training.
        """
        self.model_params = model_params or {}
        self.logger = logger

        # Handle optional early stopping configuration
        self.early_stopping_enabled = self.model_params.get("early_stopping", {}).get("enabled", False)
        self.early_stopping_params = {
            "patience": self.model_params.get("early_stopping", {}).get("patience", 5),
            "delta": self.model_params.get("early_stopping", {}).get("delta", 1e-4),
            "checkpoint_path": self.model_params.get("early_stopping", {}).get("checkpoint_path", "checkpoints/best_model.pt")
        }

        if self.early_stopping_enabled:
            self.logger.info(f"Early stopping enabled with params: {self.early_stopping_params}")

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Train the model using training data and optionally validation data.

        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target values for training.
            X_val (Optional[pd.DataFrame]): Validation feature matrix.
            y_val (Optional[pd.Series]): Validation target values.
        """
        raise NotImplementedError("Subclasses must implement 'train'")


    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Serialize the model to disk.

        Args:
            path (str): Destination path to save model.
        """
        raise NotImplementedError("Subclasses must implement 'save_model'")

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load a serialized model from disk.

        Args:
            path (str): Path to load model from.
        """
        raise NotImplementedError("Subclasses must implement 'load_model'")

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using features X.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        raise NotImplementedError("Subclasses must implement 'predict'")
    def get_params(self) -> Dict[str, Any]:
        """
        Return the model's configuration parameters.

        Returns:
            Dict[str, Any]: Model hyperparameters and configurations.
        """
        return self.model_params.copy()
