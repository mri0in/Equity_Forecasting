"""
Train a LightGBM meta-learner on combined meta-features
(oof predictions + handcrafted features).
"""

import os
import joblib
import lightgbm as lgb
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetaModelTrainer:
    """
    Trains a LightGBM model on meta-features for ensemble learning.
    """

    def __init__(
        self,
        data_path: str,
        target_col: str = "target",
        model_params: Optional[dict] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """
        Initialize the trainer with dataset path and model configuration.

        Args:
            data_path (str): Path to CSV containing meta-features.
            target_col (str): Name of the target column.
            model_params (dict, optional): LightGBM hyperparameters.
            test_size (float): Proportion of data to use for validation.
            random_state (int): Random seed for reproducibility.
        """
        self.data_path = data_path
        self.target_col = target_col
        self.model_params = model_params or {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1
        }
        self.test_size = test_size
        self.random_state = random_state
        self.model = None

    def load_data(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load meta-feature data and split into train/validation sets.

        Returns:
            Tuple: X_train, y_train, X_val, y_val
        """
        df = pd.read_csv(self.data_path)
        logger.info("Meta-feature data loaded with shape %s", df.shape)

        if self.target_col not in df.columns:
            msg = f"Target column '{self.target_col}' not found."
            logger.error(msg)
            raise ValueError(msg)

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        logger.info("Data split: train=%s, val=%s", X_train.shape, X_val.shape)

        return X_train, y_train, X_val, y_val

    def train(self) -> None:
        """
        Train the LightGBM model on the meta-feature dataset.
        """
        X_train, y_train, X_val, y_val = self.load_data()

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        logger.info("Training LightGBM meta-model...")
        self.model = lgb.train(
            self.model_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=100
        )

        y_pred = self.model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        logger.info("Validation RMSE: %.4f", rmse)

    def save_model(self, path: str) -> None:
        """
        Save the trained LightGBM model to disk.

        Args:
            path (str): Destination path (.pkl).
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call `train()` first.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Meta-model saved to %s", path)
