# src/pipeline/E_model_trainer_pipeline.py
"""
Pipeline E — Base Model Training (Non-Sequential Models Only)

Responsibilities
----------------
- Load pooled dataset produced by Pipeline D
- Train multiple base (non-LSTM) models using fixed hyperparameters
- Persist trained model artifacts under:
    datalake/runs/{run_id}/models/{model_name}/
- Emit a trained-model manifest for downstream Pipeline F
- Log all stages via TrainingMonitor

Explicit Non-Responsibilities
-----------------------------
- No hyperparameter optimization
- No LSTM / sequence models
- No ensembling or stacking
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.monitoring.monitor import TrainingMonitor
from src.utils.logger import get_logger
from src.utils.model_utils import load_config

logger = get_logger(__name__)


class ModelTrainerPipeline:
    """
    Pipeline E — trains multiple fixed-parameter base models
    on pooled dataset produced by Pipeline D.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, *, run_id: str, config_path: str) -> None:
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"E config not found: {self.config_path}")

        self.run_id = run_id

        # -----------------------------
        # Load YAML config
        # -----------------------------
        self.config = load_config(config_path)

        if not self.run_id:
            raise ValueError("run_id must be provided")

        self.train_cfg = self.config.get("training", {})

        self.models_cfg: Dict[str, dict] = self.train_cfg.get("models", {})

        if not self.models_cfg:
            raise ValueError("No models defined in E training config")


        # -----------------------------
        # Run directories
        # -----------------------------
        self.base_run_dir = Path("datalake") / "runs" / self.run_id
        self.training_dir = self.base_run_dir / "training"
        self.models_dir = self.base_run_dir / "models"

        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Dataset paths (from Pipeline D)
        # -----------------------------
        self.dataset_dir = self.base_run_dir / "dataset"
        self.X_path = self.dataset_dir / "X_pooled.parquet"
        self.y_path = self.dataset_dir / "y_pooled.parquet"

        if not self.X_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(
                "Pooled dataset missing. "
                "Expected Pipeline D outputs under "
                f"{self.dataset_dir}"
            )

        # -----------------------------
        # Runtime monitor
        # -----------------------------
        self.monitor = TrainingMonitor(
            run_id=run_id,
            save_dir=self.training_dir,
            artifact_policy="training",
            enable_plots=False,
        )

        logger.info(
            "[TRN] Initialized Pipeline E | run_id=%s | num_models=%d",
            self.run_id,
            len(self.models_cfg),
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_pooled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load pooled dataset produced by Pipeline D.

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features), dtype=float32
        y : np.ndarray
            Target vector of shape (n_samples,), dtype=float32
        """
        self.monitor.log_stage_start("load_pooled_data")

        # -----------------------------
        # Load from disk
        # -----------------------------
        X_df = pd.read_parquet(self.X_path)
        y_df = pd.read_parquet(self.y_path)

        # -----------------------------
        # Convert to NumPy (single contract)
        # -----------------------------
        X = X_df.to_numpy(dtype=np.float32, copy=True)
        y = y_df.to_numpy(dtype=np.float32, copy=True).reshape(-1)

        # -----------------------------
        # Defensive validation
        # -----------------------------
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape={X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got shape={y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Row mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}"
            )

        logger.info(
            "[TRN] Loaded pooled dataset | X=%s | y=%s | dtype=%s",
            X.shape,
            y.shape,
            X.dtype,
        )

        self.monitor.log_stage_end("load_pooled_data", {"status": "success"})

        return X, y


    # ------------------------------------------------------------------
    # Model instantiation
    # ------------------------------------------------------------------
    def _instantiate_model(self, model_name: str):
        model_cfg = self.models_cfg[model_name]
        module_path = model_cfg["module"]
        class_name = model_cfg["class"]
        params = model_cfg.get("params", {})

        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)

        logger.info(
            "[TRN] Instantiating model | name=%s | class=%s",
            model_name,
            class_name,
        )

        return model_cls(model_params={"params": params})

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:

        self.logger.info("[E] Starting modeltrainer pipeline | run_id=%s", self.run_id)

        self.monitor.log_stage_start(
            "E_training_pipeline",
            {"num_models": len(self.models_cfg)},
        )

        X, y = self._load_pooled_data()

        trained_models_manifest = []

        for model_name, model_cfg in self.models_cfg.items():
            stage = f"train_{model_name}"

            self.monitor.log_stage_start(stage, {"model": model_name})
            logger.info("[TRN] Training base model: %s", model_name)

            model = self._instantiate_model(model_name)

            # -----------------------------
            # Train
            # -----------------------------
            model.train(X, y)

            # -----------------------------
            # Persist
            # -----------------------------
            model_dir = self.models_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / "model.bin"
            model.save_model(model_path)

            logger.info("[TRN] Model persisted | %s -> %s", model_name, model_path)

            trained_models_manifest.append(
                {
                    "name": model_name,
                    "module": model_cfg["module"],
                    "class": model_cfg["class"],
                    "artifact_path": str(model_path),
                }
            )

            self.monitor.log_stage_end(stage, {"status": "success"})

        # -----------------------------
        # Persist manifest (for Pipeline F)
        # -----------------------------
        manifest_path = self.training_dir / "trained_models.json"
        manifest_path.write_text(
            json.dumps(trained_models_manifest, indent=2)
        )

        logger.info("[E] Modeltrainer pipeline completed | Trained-model manifest saved -> %s", manifest_path)
        self.monitor.log_stage_end(
            "E_modeltrainer_pipeline",
            {"status": "completed"},
        )

