"""
Pipeline E — Base Model Training (Non-LSTM)

Responsibilities:
- Load pooled dataset produced by Pipeline D
- Train multiple base models defined in config
- Save trained models under datalake/runs/{run_id}/models/
- Log all stages via TrainingMonitor

IMPORTANT:
- Does NOT perform hyperparameter optimization
- Does NOT train LSTM
- Does NOT perform ensembling
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import yaml

from src.monitoring.monitor import TrainingMonitor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainerPipeline:
    """
    Pipeline E — trains multiple non-LSTM base models
    on pooled dataset produced by Pipeline D.
    """

    def __init__(self, *, run_id: str, config_path: str) -> None:
        if not run_id:
            raise ValueError("run_id must be provided")

        self.run_id = run_id
        self.config_path = config_path

        # -----------------------------
        # Load config
        # -----------------------------
        with open(config_path, "r") as fh:
            self.config: Dict = yaml.safe_load(fh)

        train_cfg = self.config.get("training", {})
        self.models_cfg: List[Dict] = self.config.get("models", [])

        if not self.models_cfg:
            raise ValueError("No models defined in E training config")

        # -----------------------------
        # Run directories
        # -----------------------------
        self.base_run_dir = Path("datalake") / "runs" / run_id
        self.training_dir = self.base_run_dir / "training"
        self.models_dir = self.base_run_dir / "models"

        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Dataset paths (from D)
        # -----------------------------
        self.dataset_dir = self.base_run_dir / "dataset"
        self.X_path = self.dataset_dir / "X_pooled.parquet"
        self.y_path = self.dataset_dir / "y_pooled.parquet"

        if not self.X_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(
                "Pooled dataset not found. "
                "Expected Pipeline D outputs at "
                f"{self.dataset_dir}"
            )

        # -----------------------------
        # Runtime monitor
        # -----------------------------
        self.monitor = TrainingMonitor(
            run_id=run_id,
            save_dir=self.training_dir,
            visualize=bool(train_cfg.get("visualize", False)),
            flush_every=int(train_cfg.get("flush_every", 1)),
        )

        logger.info(
            "Initialized ETrainingPipeline | run_id=%s | models=%d",
            run_id,
            len(self.models_cfg),
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_pooled_data(self) -> tuple[np.ndarray, np.ndarray]:
        self.monitor.log_stage_start("load_pooled_data")

        X_df = pd.read_parquet(self.X_path)
        y_df = pd.read_parquet(self.y_path)

        X = X_df.values.astype("float32")
        y = y_df.values.astype("float32").reshape(-1)

        logger.info("Loaded pooled data | X=%s y=%s", X.shape, y.shape)
        self.monitor.log_stage_end("load_pooled_data", {"status": "success"})

        return X, y

    # ------------------------------------------------------------------
    # Model utilities
    # ------------------------------------------------------------------
    def _instantiate_model(self, model_cfg: Dict):
        module_path = model_cfg["module"]
        class_name = model_cfg["class"]
        params = model_cfg.get("params", {})

        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)

        logger.info("Instantiated model %s from %s", class_name, module_path)
        return model_cls(params)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.monitor.log_stage_start(
            "E_training_pipeline",
            {"num_models": len(self.models_cfg)},
        )

        X, y = self._load_pooled_data()

        trained_models_meta = []

        for model_cfg in self.models_cfg:
            model_name = model_cfg["name"]
            stage = f"train_{model_name}"

            self.monitor.log_stage_start(stage, {"model": model_name})
            logger.info("Training model: %s", model_name)

            model = self._instantiate_model(model_cfg)

            # -----------------------------
            # Train
            # -----------------------------
            model.train(X, y)

            # -----------------------------
            # Save
            # -----------------------------
            model_dir = self.models_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / "model.bin"
            model.save(model_path)

            logger.info("Saved model %s -> %s", model_name, model_path)

            trained_models_meta.append(
                {
                    "name": model_name,
                    "module": model_cfg["module"],
                    "class": model_cfg["class"],
                    "path": str(model_path),
                }
            )

            self.monitor.log_stage_end(stage, {"status": "success"})

        # -----------------------------
        # Persist manifest for downstream F
        # -----------------------------
        manifest_path = self.training_dir / "trained_models.json"
        manifest_path.write_text(json.dumps(trained_models_meta, indent=2))

        logger.info("Saved trained models manifest -> %s", manifest_path)

        self.monitor.log_stage_end("E_training_pipeline", {"status": "completed"})
