"""
Generate out-of-fold (OOF) predictions from base models (e.g., LSTM), 
using walk-forward validation (wfv) method which are 
to be used as meta-features in the ensemble meta-model.

This script performs walk-forward validation, collects OOF predictions,
and saves them to disk.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from src.models.lstm_model import LSTMModel
from src.utils.logger import get_logger
from sklearn.model_selection import TimeSeriesSplit
import os

logger = get_logger(__name__)


class OOFGenerator:
    """
    Class to generate out-of-fold predictions for a given model using walk-forward validation.
    """

    def __init__(self, model_params: dict, n_splits: int = 5) -> None:
        """
        Initialize the OOF generator.

        Args:
            model_params (dict): Parameters for the base model (e.g., LSTM).
            n_splits (int): Number of walk-forward validation splits.
        """
        self.model_params = model_params
        self.n_splits = n_splits

    def load_data(self, x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load input features and targets.

        Args:
            x_path (str): Path to X.npy
            y_path (str): Path to y.npy

        Returns:
            Tuple of input and target arrays.
        """
        try:
            X = np.load(x_path)
            y = np.load(y_path)
            logger.info("Loaded data for OOF generation: X=%s, y=%s", X.shape, y.shape)
            return X, y
        except Exception as e:
            logger.exception("Failed to load data for OOF generation: %s", str(e))
            raise

    def generate(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate OOF predictions and corresponding true values.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values

        Returns:
            Tuple[np.ndarray, np.ndarray]: OOF predictions and true values
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof_preds = []
        oof_targets = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info("Processing fold %d", fold + 1)

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = LSTMModel(model_params=self.model_params)
            model.train(X_train, y_train)

            preds = model.predict(X_val)

            oof_preds.append(preds)
            oof_targets.append(y_val)

        # Concatenate all folds
        return np.concatenate(oof_preds), np.concatenate(oof_targets)

    def save_oof(self, preds: np.ndarray, targets: np.ndarray, out_dir: str) -> None:
        """
        Save OOF predictions and targets.

        Args:
            preds (np.ndarray): OOF predictions
            targets (np.ndarray): OOF targets
            out_dir (str): Directory to save files
        """
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "oof_preds.npy"), preds)
        np.save(os.path.join(out_dir, "oof_targets.npy"), targets)
        logger.info("Saved OOF predictions and targets to %s", out_dir)
