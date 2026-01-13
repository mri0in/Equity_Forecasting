# src/models/lightGBM_model.py
"""
LightGBM Gradient Boosting Model

Risk Axis Covered
-----------------
- Non-linear interaction risk
- Threshold and saturation effects
- Conditional feature dependencies

What This Model Hedges Against
------------------------------
- Missed alpha from non-linear relationships
- Feature interactions not expressible in linear models
- Regime-specific conditional behavior (e.g., volatility × momentum)

Failure Modes / What It Does NOT Capture
----------------------------------------
- Long temporal dependencies unless explicitly encoded
- Extreme extrapolation beyond training distribution
- Very low-signal environments (may overfit noise)

Role in Ensemble
----------------
Primary non-linear alpha engine.
Typically dominates performance in heterogeneous, feature-rich regimes.
Benefits strongly from pooled cross-equity data.
"""


from typing import Optional
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.models.base_model import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM regressor for non-linear tabular modeling.
    """

    def __init__(self, model_params: Optional[dict] = None) -> None:
        model_params = model_params or {}
        super().__init__(model_params)
        self.params = self.model_params.get("params", {})
        self.model: Optional[lgb.LGBMRegressor] = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        self.logger.info("Training LightGBM model")

        self.model = lgb.LGBMRegressor(**self.params)

        if self.early_stopping_enabled and X_val is not None:
            self.model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.early_stopping_params["patience"]
                    )
                ],
            )
        else:
            self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        joblib.dump(self.model, path)
        self.logger.info("LightGBM model saved to %s", path)

    def load_model(self, path: str) -> None:
        self.model = joblib.load(path)
        self.logger.info("LightGBM model loaded from %s", path)
