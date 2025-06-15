from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class  BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    Defines the unified interface and shared model parameter handling.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize model with parameters and setup logging.

        Args:
            model_params (Optional[Dict[str, Any]]): Configuration dictionary for model and training.
        """
        self.model_params = model_params or {}
        self.logger = logger

        # Handle early stopping defaults
        self.early_stopping_enabled = self.model_params.get("early_stopping", {}).get("enabled", False)
        self.early_stopping_params = {
            "patience": self.model_params.get("early_stopping", {}).get("patience", 5),
            "delta": self.model_params.get("early_stopping", {}).get("delta", 1e-4),
            "checkpoint_path": self.model_params.get("early_stopping", {}).get("checkpoint_path", "checkpoints/best_model.pt")
        }

        if self.early_stopping_enabled:
            self.logger.info(f"Early stopping enabled with params: {self.early_stopping_params}")

    @abstractmethod
    def train(self,
              X: pd.DataFrame,
              y: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> None:
        """Train the model using features X and target y. Optional validation set for early stopping."""
        ... # pragma: no cover

    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using features X."""
        ... # pragma: no cover

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return the hyperparameters/config used by this model."""
        ... # pragma: no cover

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize the model to disk."""
        ... # pragma: no cover

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a serialized model from disk."""
        ... # pragma: no cover
