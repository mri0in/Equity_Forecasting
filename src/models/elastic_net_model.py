# src/models/elastic_net_model.py
"""
Elastic Net Regression Model

Risk Axis Covered
-----------------
- Feature sparsity risk
- Redundant signal inflation
- Over-parameterization in wide feature sets

What This Model Hedges Against
------------------------------
- Irrelevant or weak predictors in large technical/fundamental feature spaces
- Correlated signals where Lasso alone would be unstable
- Feature drift where only a subset remains informative

Failure Modes / What It Does NOT Capture
----------------------------------------
- Strong non-linear interactions
- High-frequency regime changes
- Long-memory temporal dependencies

Role in Ensemble
----------------
Functions as a sparse signal selector.
Performs well when alpha is driven by a few dominant features.
Often outperforms Ridge during transitional regimes with feature churn.
"""

from typing import Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from src.models.base_model import BaseModel


class ElasticNetModel(BaseModel):
    """
    Elastic Net regression for sparse and correlated features.
    """

    def __init__(self, model_params: Optional[dict] = None) -> None:
        super().__init__(model_params)
        self.alpha = model_params.get("alpha", 1.0)
        self.l1_ratio = model_params.get("l1_ratio", 0.5)
        self.model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=10_000,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        self.logger.info(
            "Training ElasticNet | alpha=%s | l1_ratio=%s",
            self.alpha,
            self.l1_ratio,
        )
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        joblib.dump(self.model, path)
        self.logger.info("ElasticNet model saved to %s", path)

    def load_model(self, path: str) -> None:
        self.model = joblib.load(path)
        self.logger.info("ElasticNet model loaded from %s", path)
