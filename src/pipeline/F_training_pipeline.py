# src/pipeline/F_training_pipeline.py
"""
Pipeline module to orchestrate end-to-end model training.

This script performs:
1. Config loading from model_utils
2. Dynamic model instantiation
3. Training data loadingl
4. Model training
5. Saving trained model

⚠️ IMPORTANT WARNING:
Do NOT call these classes directly in end-user workflows.
Use wrappers in src/pipeline/pipeline_wrapper.py to enforce
orchestration, logging, retries, and task markers.
"""

from datetime import datetime, timezone
import numpy as np
from typing import Tuple
from src.utils.logger import get_logger
from src.utils.model_utils import load_config_and_model
from src.monitoring.monitor import TrainingMonitor

logger = get_logger(__name__)


class ModelTrainerPipeline:
    """
    Pipeline class for training any model defined via configuration.
    """

    def __init__(self, config_path: str):
        """
        Constructor: loads configuration, model instance, and initializes monitoring.

        Args:
            config_path (str): Path to YAML config file
        """
        if not config_path:
            raise ValueError("config_path must be provided to ModelTrainerPipeline")

        
        self.config, self.model = load_config_and_model(config_path)
        train_cfg = self.config.get("training", {})

        scope = train_cfg.get("scope", "GLOBAL")  # GLOBAL / NIFTY / RELIANCE / etc
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id: str = f"{scope}_TRAIN_{timestamp}"

        # -------------------------------------------------
        # Run directory (monitor artifacts)
        # -------------------------------------------------
        base_run_dir = train_cfg.get("run_dir", "runs/training")
        self.run_dir: str = f"{base_run_dir}/{self.run_id}"

        # -------------------------------------------------
        # Training monitor (MUST be runtime-scoped)
        # -------------------------------------------------
        self.monitor = TrainingMonitor(
            run_id=self.run_id,
            save_dir=self.run_dir,
            visualize=bool(train_cfg.get("visualize", False)),
            flush_every=int(train_cfg.get("flush_every", 1)),
        )

        logger.info(
            "Initialized ModelTrainerPipeline | run_id=%s | run_dir=%s",
            self.run_id,
            self.run_dir,
        )

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training features and targets from .npy files.

        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features), y (targets)
        """
        self.monitor.log_stage_start("load_training_data")
        try:
            x_path = self.config["data"]["X_train_path"]
            y_path = self.config["data"]["y_train_path"]
            X = np.load(x_path)
            y = np.load(y_path)
            logger.info("Loaded training data: X=%s, y=%s", X.shape, y.shape)
        except Exception as e:
            logger.exception("Error loading training data")
            self.monitor.log_stage_end("load_training_data", success=False)
            raise
        self.monitor.log_stage_end("load_training_data")
        return X, y

    def run(self) -> None:
        """
        Run the training pipeline:
        - Load data
        - Train model
        - Save model
        """
        self.monitor.log_stage_start("run_training_pipeline")
        try:
            X, y = self.load_training_data()
            self.model.train(X, y)

            save_path = self.config["training"]["save_path"]
            self.model.save_model(save_path)
            logger.info("Model training completed and saved to: %s", save_path)
        except Exception as e:
            logger.exception("Training pipeline failed")
            self.monitor.log_stage_end("run_training_pipeline", success=False)
            raise
        self.monitor.log_stage_end("run_training_pipeline")

