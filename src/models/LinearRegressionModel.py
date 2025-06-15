# src/models/linear_model.py

from typing import Any, Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LinearRegressionModel(BaseModel):
    """
    Linear regression model wrapper for scikit-learn.
    """

    def __init__(self, **params: Any):
        # Validate params
        if params.get("fit_intercept", True) not in [True, False]:
            raise ValueError("fit_intercept must be a boolean")
        if params.get("normalize", False) not in [True, False]:
            raise ValueError("normalize must be a boolean")

        # Store config
        self._params: Dict[str, Any] = params.copy()
        # Instantiate model
        self.model = LinearRegression(**params)
        logger.info("Initialized LinearRegressionModel with params: %s", params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info("Training on %d samples", len(X))
        self.model.fit(X, y)
        logger.info("Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        logger.info("Predicting %d samples", len(X))
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return self._params.copy()

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"model": self.model, "params": self._params}, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        import joblib
        payload = joblib.load(path)
        self.model = payload["model"]
        self._params = payload["params"]
        logger.info("Model loaded from %s", path)
