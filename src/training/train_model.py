# src/models/train_model.py
"""
train_model.py

Pure training engine for equity forecasting models.

Responsibilities:
- Load runtime YAML config (already resolved by pipeline E)
- Load training / optional validation data from disk
- Dynamically import and initialize model
- Train model (with optional early stopping)
- Save trained model to injected save_path
- Emit stage-level telemetry to TrainingMonitor
"""

from __future__ import annotations

import importlib
from typing import Tuple, Type, Optional

import yaml
import numpy as np

from src.utils.logger import get_logger
from src.monitoring.monitor import TrainingMonitor

logger = get_logger(__name__)


class ModelTrainer:
    """
    Stateless training engine.
    All paths and parameters must be injected via config.
    """

    def __init__(self, *, config_path: str, monitor: TrainingMonitor):
        if not config_path:
            raise ValueError("config_path must be provided")

        self.config_path = config_path
        self.monitor = monitor

        self.config = self._load_config()
        self.model = self._initialize_model()

        # Cache data after first load
        self._data_cache = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _load_config(self) -> dict:
        self.monitor.log_stage_start("load_config", {"config_path": self.config_path})
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info("Loaded training config from %s", self.config_path)
            return config
        finally:
            self.monitor.log_stage_end("load_config")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def load_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load training and optional validation data.
        """
        self.monitor.log_stage_start("load_data")
        try:
            if self._data_cache is not None:
                return self._data_cache

            data_cfg = self.config["data"]

            X_train = np.load(data_cfg["X_train_path"])
            y_train = np.load(data_cfg["y_train_path"])

            X_val = y_val = None

            early_stopping_cfg = (
                self.config
                .get("model", {})
                .get("params", {})
                .get("early_stopping", {})
            )

            if early_stopping_cfg.get("enabled", False):
                X_val = np.load(data_cfg["X_val_path"])
                y_val = np.load(data_cfg["y_val_path"])
                logger.info(
                    "Loaded validation data | X_val=%s y_val=%s",
                    X_val.shape,
                    y_val.shape,
                )

            logger.info(
                "Loaded training data | X_train=%s y_train=%s",
                X_train.shape,
                y_train.shape,
            )

            self._data_cache = (X_train, y_train, X_val, y_val)
            return self._data_cache

        finally:
            self.monitor.log_stage_end("load_data")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _initialize_model(self):
        self.monitor.log_stage_start("initialize_model")
        try:
            model_cfg = self.config["model"]

            module_path = model_cfg["module"]
            class_name = model_cfg["class"]
            params = model_cfg.get("params", {})

            module = importlib.import_module(module_path)
            model_class: Type = getattr(module, class_name)

            logger.info(
                "Initialized model | class=%s module=%s",
                class_name,
                module_path,
            )

            return model_class(params)

        finally:
            self.monitor.log_stage_end("initialize_model")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(self) -> None:
        """
        Train the model. Does NOT save artifacts.
        """
        self.monitor.log_stage_start("train")
        try:
            X, y, X_val, y_val = self.load_data()
            self.model.train(X, y, X_val, y_val)
            logger.info("Model training completed")
        finally:
            self.monitor.log_stage_end("train")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self) -> None:
        """
        Persist trained model to disk.
        """
        self.monitor.log_stage_start("save")
        try:
            save_path = self.config["training"]["save_path"]
            self.model.save(save_path)
            logger.info("Model saved to %s", save_path)
        finally:
            self.monitor.log_stage_end("save")
