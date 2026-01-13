"""
Pipeline F — Global Inference Pipeline

Purpose
-------
Runs inference using trained global models produced by Pipeline E
on the pooled dataset (X_pooled.parquet).

Contracts (STRICT)
------------------
1. Consumes:
   - runs/{run_id}/dataset/X_pooled.parquet
   - runs/{run_id}/training/trained_models.json

2. Produces:
   - runs/{run_id}/inference/
       ├── predictions/
       │     └── <model_name>_preds.npy
       └── metrics.jsonl

3. NEVER scans model directories
4. Loads models ONLY via trained_models.json
5. Artifact path always points to model ROOT directory
"""

import importlib
import os
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.monitoring.monitor import TrainingMonitor
from src.training.evaluate import compute_metrics
from src.utils.model_utils import load_config

logger = get_logger(__name__)


class InferencePipeline:
    """
    Pipeline F — Inference

    Runs inference using all trained global models on pooled features.
    """

    def __init__(self, run_id: str, config_path: str):
        if not run_id:
            raise ValueError("run_id must be provided")
        if not config_path:
            raise ValueError("config_path must be provided")

        self.run_id = run_id
        self.config_path = config_path

        self.config = load_config(config_path)

        self.base_run_dir = Path("datalake") / "runs" / self.run_id
        self.dataset_dir = self.base_run_dir / "dataset"
        self.training_dir = self.base_run_dir / "training"
        self.inference_dir = self.base_run_dir / "inference"
        self.pred_dir = self.inference_dir / "predictions"

        os.makedirs(self.pred_dir, exist_ok=True)

        self.monitor = TrainingMonitor(
            run_id=run_id,
            save_dir=self.inference_dir,
            artifact_policy="none",
            enable_plots=False,
        )
        
        self.logger = logger

    # ------------------------------------------------------------------
    # Load pooled dataset
    # ------------------------------------------------------------------
    def _load_features(self) -> np.ndarray:
        X_path = self.dataset_dir / "X_pooled.parquet"
        if not X_path.exists():
            raise FileNotFoundError(f"Missing pooled features: {X_path}")

        df = pd.read_parquet(X_path)
        logger.info("Loaded pooled features with shape %s", df.shape)
        return df.values

    # ------------------------------------------------------------------
    # Load trained models registry
    # ------------------------------------------------------------------
    def _load_model_registry(self) -> List[Dict]:
        registry_path = self.training_dir / "trained_models.json"

        if not registry_path.exists():
            raise FileNotFoundError(
            f"trained_models.json not found at {registry_path}"
            )

        with open(registry_path, "r") as f:
            registry = json.load(f)

        logger.info("Loaded %d trained models", len(registry))
        return registry

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute inference for all trained models.

        """
        self.logger.info("[F] Starting inference pipeline | run_id=%s", self.run_id)
        self.monitor.log_stage_start("F_inference_pipeline")

        X = self._load_features()
        model_registry = self._load_model_registry()

        y_path = self.dataset_dir / "y_pooled.parquet"
        if not y_path.exists():
            raise FileNotFoundError(f"Missing pooled targets: {y_path}")

        # save true targets for reference
        y_true = pd.read_parquet(y_path).values.ravel()
        y_true_path = self.inference_dir / "y_true.npy"
        np.save(y_true_path, y_true)    
        logger.info("Saved true targets to %s", y_true_path)

        results: Dict[str, Dict[str, float]] = {}

        self.logger.info("Loaded model registry with %d models", len(model_registry))

        for model_meta in model_registry:
            name = model_meta["name"]
            module = model_meta["module"]
            class_name = model_meta["class"]
            artifact_path = model_meta["artifact_path"]

            self.monitor.log_stage_start(
                "model_inference",
                {"model": name},
            )

            logger.info("[INF] Running inference for model: %s", name)

            module = importlib.import_module(module)
            model_cls = getattr(module, class_name)
            model = model_cls.load_model(Path(artifact_path))

            y_pred = model.predict(X) 
            pred_path = self.pred_dir / f"{name}_preds.npy"

            np.save(pred_path, y_pred)

            logger.info(
                "[INF] Saved predictions for %s to %s", name, pred_path
            )


            self.monitor.log_stage_end(
                "model_inference",
                {"model": name, "status": "success"},
            )

        success_flag = os.path.join(self.inference_dir, "_SUCCESS")
        with open(success_flag, "w") as f:
            f.write("")
            
        self.monitor.log_stage_end(
            "F_inference_pipeline", {"status": "completed"}
        )
        logger.info("[F] Inference pipeline completed successfully")

