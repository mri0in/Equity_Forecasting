# src/pipeline/G_wfv_pipeline.py

"""
Pipeline G — Walk-Forward Validation (WFV)

Purpose
-------
Pipeline G performs temporal robustness validation of trained global models
using walk-forward (rolling / expanding window) evaluation on pooled equity data.

This pipeline exists to ensure that only models with stable, regime-robust
performance are allowed to influence downstream global signals and adapters.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.model_utils import load_config
from src.training.evaluate import compute_metrics

logger = get_logger(__name__)


class WalkForwardValidationPipeline:
    """
    Pipeline G — Walk-Forward Validation

    Validates temporal robustness of trained global models using
    rolling / expanding windows on pooled equity data.
    """

    def __init__(self, run_id: str, config_path: str) -> None:
        if not run_id:
            raise ValueError("run_id must be provided")
        if not config_path:
            raise ValueError("config_path must be provided")

        self.run_id = run_id
        self.config = load_config(config_path)

        self.base_run_dir = Path("datalake") / "runs" / self.run_id
        self.dataset_dir = self.base_run_dir / "dataset"
        self.training_dir = self.base_run_dir / "training"
        self.output_dir = self.base_run_dir / "walk_forward"

        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logger

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        X_path = self.dataset_dir / "X_pooled.parquet"
        y_path = self.dataset_dir / "y_pooled.parquet"

        if not X_path.exists():
            raise FileNotFoundError(f"Missing {X_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Missing {y_path}")

        X = pd.read_parquet(X_path).values
        y = pd.read_parquet(y_path).values.ravel()

        if len(X) != len(y):
            raise ValueError("X and y length mismatch")

        self.logger.info("Loaded pooled dataset: X=%s y=%s", X.shape, y.shape)
        return X, y

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------
    def _load_model_registry(self) -> List[Dict]:
        registry_path = self.training_dir / "trained_models.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"Missing {registry_path}")

        with open(registry_path, "r") as f:
            registry = json.load(f)

        if not registry:
            raise ValueError("trained_models.json is empty")

        self.logger.info("Loaded %d trained models", len(registry))
        return registry

    # ------------------------------------------------------------------
    # Window generator
    # ------------------------------------------------------------------
    def _generate_windows(self, n_samples: int) -> List[Tuple[int, int]]:
        """
        Generate walk-forward windows.

        Uses expanding train window with fixed validation horizon.
        """
        window_cfg = self.config["walk_forward"]
        train_min = window_cfg["min_train_size"]
        val_size = window_cfg["val_size"]
        step = window_cfg["step_size"]

        windows = []
        start = train_min

        while start + val_size <= n_samples:
            windows.append((start, start + val_size))
            start += step

        if not windows:
            raise ValueError("No walk-forward windows generated")

        self.logger.info("Generated %d walk-forward windows", len(windows))
        return windows

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.logger.info("[G] Starting Walk-Forward Validation | run_id=%s", self.run_id)

        X, y = self._load_dataset()
        model_registry = self._load_model_registry()
        windows = self._generate_windows(len(y))

        metrics_path = self.output_dir / "per_model_window_metrics.jsonl"
        stability_path = self.output_dir / "stability_scores.json"
        eligible_path = self.output_dir / "eligible_models.json"

        all_metrics: List[Dict] = []

        # --------------------------------------------------------------
        # Walk-forward evaluation
        # --------------------------------------------------------------
        for model_meta in model_registry:
            name = model_meta["name"]
            module = model_meta["module"]
            class_name = model_meta["class"]
            artifact_path = Path(model_meta["artifact_path"])

            self.logger.info("[G] Evaluating model: %s", name)

            module_obj = __import__(module, fromlist=[class_name])
            model_cls = getattr(module_obj, class_name)
            model = model_cls() 
            model.load_model(artifact_path)

            for train_end, val_end in windows:
                X_val = X[train_end:val_end]
                y_val = y[train_end:val_end]

                y_pred = model.predict(X_val)

                # --- Shape normalization  ---
                if y_pred.ndim == 2 and y_pred.shape[1] == 1:
                    y_pred = y_pred.ravel()

                if y_pred.ndim != 1:
                    raise ValueError(
                        f"Invalid prediction shape from model {name}: {y_pred.shape}"
                    )

                metrics = compute_metrics(y_val, y_pred)
                metrics_record = {
                    "model": name,
                    "window_start": int(train_end),
                    "window_end": int(val_end),
                    **metrics,
                }
                all_metrics.append(metrics_record)

        # --------------------------------------------------------------
        # Persist window metrics
        # --------------------------------------------------------------
        if not all_metrics:
            raise RuntimeError("No walk-forward metrics computed")

        pd.DataFrame(all_metrics).to_json(
            metrics_path, orient="records", lines=True
        )
        self.logger.info("[WFV] Saved window metrics to %s", metrics_path)

        # --------------------------------------------------------------
        # Stability aggregation
        # --------------------------------------------------------------
        df = pd.DataFrame(all_metrics)

        grouped = df.groupby("model")

        agg_dict = {
            "mean_rmse": ("rmse", "mean"),
            "std_rmse": ("rmse", "std"),
        }

        if "mae" in df.columns:
            agg_dict["mean_mae"] = ("mae", "mean")

        if "r2" in df.columns:
            agg_dict["mean_r2"] = ("r2", "mean")

        agg = grouped.agg(**agg_dict).reset_index()

        agg["stability_score"] = 1.0 / (1.0 + agg["std_rmse"])

        with open(stability_path, "w") as f:
            json.dump(agg.to_dict(orient="records"), f, indent=2)

        self.logger.info("[WFV] Saved stability scores to %s", stability_path)

        # --------------------------------------------------------------
        # Eligibility filtering
        # --------------------------------------------------------------
        cfg = self.config["walk_forward"]
        rmse_thresh = cfg["max_mean_rmse"]
        stability_thresh = cfg["min_stability_score"]

        eligible = agg[
            (agg["mean_rmse"] <= rmse_thresh)
            & (agg["stability_score"] >= stability_thresh)
        ]

        if eligible.empty:
            self.logger.error(
                "[G] No eligible models. Summary:\n%s",
                agg.sort_values("mean_rmse")
            )
            raise RuntimeError("No eligible models after walk-forward validation")

        eligible_models = {
            "eligible_models": eligible[
                ["model", "mean_rmse", "stability_score"]
            ].to_dict(orient="records")
        }

        with open(eligible_path, "w") as f:
            json.dump(eligible_models, f, indent=2)

        self.logger.info("[WFV] Saved eligible models are %s", eligible_models)
        self.logger.info("[G] Walk-Forward Pipeline completed successfully")
