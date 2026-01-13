# src/models/ridge_regg_model.py
"""
Ridge Regression Model

Risk Axis Covered
-----------------
- Linear factor risk
- Noise amplification due to multicollinearity
- Overfitting in high-dimensional but stable feature spaces

What This Model Hedges Against
------------------------------
- Excessive coefficient variance caused by correlated technical indicators
- Instability in regimes where relationships are approximately linear
- Small-sample noise when features >> samples

Failure Modes / What It Does NOT Capture
----------------------------------------
- Non-linear feature interactions
- Regime shifts with structural breaks
- Temporal dependencies or lag dynamics beyond explicit features

Role in Ensemble
----------------
Acts as a low-variance anchor model.
Provides stability during calm or mean-reverting market regimes.
Often dominates during low-volatility, low-dispersion periods.
"""

from typing import Optional
from fastapi import params
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.base_model import BaseModel


class RidgeRegressionModel(BaseModel):
    """
    Ridge Regression model for stable linear forecasting.
    """

    def __init__(self, model_params: Optional[dict] = None) -> None:
        super().__init__(model_params)
        self.alpha = model_params.get("alpha", 1.0)
        self.model = Ridge(alpha=self.alpha)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        self.logger.info("Training Ridge Regression | alpha=%s", self.alpha)
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        joblib.dump(self.model, path)
        self.logger.info("Ridge model saved to %s", path)

    def load_model(self, path: str) -> None:
        self.model = joblib.load(path)
        self.logger.info("Ridge model loaded from %s", path)
