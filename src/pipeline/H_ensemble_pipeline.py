# src/pipeline/H_ensemble_pipeline.py

"""
Pipeline H — Global Ensemble Pipeline

Purpose
-------
Constructs a single, stable global signal by ensembling predictions
from models that have passed Walk-Forward Validation (Pipeline G).

This pipeline is the FINAL stage of the global signal layer.
Its output is consumed exclusively by the downstream Adapter, which
maps the global signal to user-fed single-equity price prediction.

CONTRACT
---------------
Consumes (ONLY):
1. runs/{run_id}/inference/predictions/<model_name>_preds.npy
2. runs/{run_id}/walk_forward/eligible_models.json
3. runs/{run_id}/walk_forward/walk_forward_summary.json

Produces:
runs/{run_id}/ensemble/
├── global_signal.npy          # Final ensemble signal (1D array)
├── ensemble_weights.json      # Model weights used in ensemble
├── metrics.jsonl              # Diagnostics and metadata
└── _SUCCESS                   # Completion marker


Design Philosophy
-----------------
Pipeline G decides *which* models are safe.
Pipeline H decides *how* to combine safe models.

The ensemble reduces variance and produces a robust global signal
suitable for downstream adaptation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils.logger import get_logger
from src.monitoring.monitor import TrainingMonitor

logger = get_logger(__name__)


class EnsemblePipeline:
    """
    Pipeline H — Global Ensemble

    Combines predictions from G-approved models into a single
    global signal using stability-weighted averaging.
    """

    def __init__(self, run_id: str):
        if not run_id:
            raise ValueError("run_id must be provided")

        self.run_id = run_id

        self.base_run_dir = Path("datalake") / "runs" / run_id

        self.inference_dir = self.base_run_dir / "inference"
        self.predictions_dir = self.inference_dir / "predictions"

        self.wfv_dir = self.base_run_dir / "walk_forward"
        self.ensemble_dir = self.base_run_dir / "ensemble"

        os.makedirs(self.ensemble_dir, exist_ok=True)

        self.monitor = TrainingMonitor(
            run_id=run_id,
            save_dir=self.ensemble_dir,
            artifact_policy="none",
            enable_plots=False,
        )

        self.logger = logger

    # ------------------------------------------------------------------
    # Load eligible models from Pipeline G
    # ------------------------------------------------------------------
    def _load_eligible_models(self) -> List[str]:
        eligible_path = self.wfv_dir / "eligible_models.json"

        if not eligible_path.exists():
            raise FileNotFoundError(
                f"eligible_models.json not found at {eligible_path}"
            )

        with open(eligible_path, "r") as f:
            data = json.load(f)

        models = data.get("eligible_models", [])
        if not models:
            raise RuntimeError("No eligible models provided by Pipeline G")

        self.logger.info(
            "[H] Loaded %d eligible models: %s",
            len(models),
            models,
        )
        return models

    # ------------------------------------------------------------------
    # Load walk-forward summary (for stability scores)
    # ------------------------------------------------------------------
    def _load_stability_scores(self) -> Dict[str, float]:
        stability_path = self.wfv_dir / "stability_scores.json"

        if not stability_path.exists():
            raise FileNotFoundError(
                f"stability_scores.json not found at {stability_path}"
            )

        with open(stability_path, "r") as f:
            stability_data = json.load(f)

        # Extract list from JSON structure (handle both list and dict with list inside)
        data_list = stability_data if isinstance(stability_data, list) else stability_data.get("stability_scores", [])
        
        scores = {
            item.get("model"): item.get("stability_score")
            for item in data_list
        }
        return scores

    # ------------------------------------------------------------------
    # Load model predictions
    # ------------------------------------------------------------------
    def _load_predictions( self, model_names: List[str] ) -> Dict[str, np.ndarray]:
        preds: Dict[str, np.ndarray] = {}

        for _model in model_names:
            model = _model.get("model", [])
            pred_path = self.predictions_dir / f"{model}_preds.npy"
            if not pred_path.exists():
                raise FileNotFoundError(
                    f"Missing predictions for model '{model}' at {pred_path}"
                )

            arr = np.load(pred_path)

            if arr.ndim != 1:
                raise ValueError(
                    f"Predictions for model '{model}' must be 1D, got shape {arr.shape}"
                )

            preds[model] = arr

        lengths = {len(v) for v in preds.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"Prediction length mismatch across models: {lengths}"
            )

        return preds

    # ------------------------------------------------------------------
    # Run ensemble
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute the global ensemble pipeline.
        """
        self.logger.info("[H] Starting ensemble pipeline | run_id=%s", self.run_id)
        self.monitor.log_stage_start("H_ensemble_pipeline")

        eligible_models = self._load_eligible_models()
        stability_scores = self._load_stability_scores()
        predictions = self._load_predictions(eligible_models)

        # ------------------------------------------------------------------
        # Compute ensemble weights (stability-weighted)
        # ------------------------------------------------------------------
        raw_weights = {}
        for m in eligible_models:
            model_name = m.get("model") if isinstance(m, dict) else m
            if not model_name:
                raise RuntimeError(f"Invalid model entry in eligible_models: {m}")
            
            score = stability_scores.get(model_name)
            if score is None:
                raise KeyError(f"Missing stability score for model '{model_name}'")
            raw_weights[model_name] = score

        weight_sum = sum(raw_weights.values())
        if weight_sum <= 0:
            raise RuntimeError("Invalid ensemble weights (sum <= 0)")

        weights = {
            model: score / weight_sum
            for model, score in raw_weights.items()
        }

        self.logger.info("[H] Ensemble weights: %s", weights)

        # ------------------------------------------------------------------
        # Compute ensemble signal
        # ------------------------------------------------------------------
        ensemble_signal = np.zeros_like(
            next(iter(predictions.values())),
            dtype=float,
        )

        for model, preds in predictions.items():
            ensemble_signal += weights[model] * preds

        # ------------------------------------------------------------------
        # Persist artifacts
        # ------------------------------------------------------------------
        signal_path = self.ensemble_dir / "global_signal.npy"
        np.save(signal_path, ensemble_signal)

        weights_path = self.ensemble_dir / "ensemble_weights.json"
        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=2)

        metrics_path = self.ensemble_dir / "metrics.jsonl"
        with open(metrics_path, "a") as f:
            record = {
                "stage": "ensemble",
                "num_models": len(eligible_models),
                "models": eligible_models,
                "weights": weights,
            }
            f.write(json.dumps(record) + "\n")

        success_flag = self.ensemble_dir / "_SUCCESS"
        with open(success_flag, "w") as f:
            f.write("")

        self.monitor.log_stage_end(
            "H_ensemble_pipeline",
            {"status": "completed", "num_models": len(eligible_models)},
        )

        self.logger.info("[H] Ensemble pipeline completed successfully")
