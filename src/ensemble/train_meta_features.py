"""
Train a LightGBM meta-learner on combined meta-features
(oof predictions + handcrafted features).
"""

import logging
import os
import joblib
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Optional
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MetaFeatureTrainer:
    """
    Trainer for meta-feature model using LightGBM.
    """

    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42,
                 model_params: Optional[dict] = None) -> None:
        """
        Initialize the trainer with dataset path and LightGBM params.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")

        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1
        }
        self.model: Optional[lgb.Booster] = None
        logger.info("Initialized MetaFeatureTrainer with data=%s", data_path)

    def prepare_datasets(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load dataset and split into train/validation sets.
        """
        logger.info("Loading meta-feature dataset from %s", self.data_path)
        data = pd.read_csv(self.data_path)

        if "target" not in data.columns:
            raise ValueError("Dataset must contain a 'target' column.")

        X = data.drop(columns=["target"])
        y = data["target"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info("Prepared datasets: train=%d, val=%d", len(X_train), len(X_val))
        return X_train, y_train, X_val, y_val

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> lgb.Booster:
        """
        Train the LightGBM model.
        """
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

        logger.info("Best iteration: %d", self.model.best_iteration)
        return self.model

    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Evaluate model using RMSE.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        y_pred = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        rmse = mean_squared_error(y_val, y_pred)
        logger.info("Validation RMSE: %.4f", rmse)

        # Feature importance logging
        importance = self.model.feature_importance(importance_type="gain")
        features_sorted = sorted(zip(X_val.columns, importance), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 features by gain: %s", features_sorted)

        return rmse

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        joblib.dump(self.model, path)
        logger.info("Meta-model saved to %s", path)
