# src/pipeline/F_ensemble_pipeline.py
"""
Pipeline F — Ensemble & Meta-Modeling

Responsibilities
----------------
- Consume base-model predictions from Pipeline E
- Apply simple ensemble strategies (mean / weighted / median)
- Optionally perform stacked ensembling (OOF → meta-features → meta-model)
- Persist ensemble artifacts under the SAME run_id

Directory Layout
----------------
datalake/runs/{run_id}/ensemble/
├── predictions/
├── oof/
└── meta/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from src.monitoring.monitor import TrainingMonitor
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.ensemble.simple_ensembler import SimpleEnsembler
from src.ensemble.generate_oof import OOFGenerator
from src.ensemble.meta_features import MetaFeaturesBuilder
from src.ensemble.train_meta_features import MetaFeatureTrainer
from src.training.evaluate import compute_metrics

logger = get_logger(__name__)


class EnsemblePipeline:
    """
    Pipeline F — Ensemble & Meta-Modeling
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, *, run_id: str, config_path: str) -> None:
        if not run_id:
            raise ValueError("run_id must be provided")

        self.run_id = run_id
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        # -----------------------------
        # Load config
        # -----------------------------
        self.config = load_config(config_path)

        self.ensemble_cfg = self.config.get("ensemble", {})

        # -----------------------------
        # Run directories (strict lineage)
        # -----------------------------
        self.base_run_dir = Path("datalake") / "runs" / self.run_id
        self.ensemble_dir = self.base_run_dir / "ensemble"
        self.pred_dir = self.ensemble_dir / "predictions"
        self.oof_dir = self.ensemble_dir / "oof"
        self.meta_dir = self.ensemble_dir / "meta"

        for d in [self.ensemble_dir, self.pred_dir, self.oof_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Monitoring
        # -----------------------------
        self.monitor = TrainingMonitor(
            run_id=self.run_id,
            save_dir=self.ensemble_dir,
            artifact_policy="none",
            enable_plots=False,
        )

        logger.info(
            "Initialized Pipeline F | run_id=%s | method=%s",
            self.run_id,
            self.ensemble_cfg.get("method", "mean"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_predictions(self, paths: List[Path]) -> List[np.ndarray]:
        self.monitor.log_stage_start(
            "load_predictions", {"num_paths": len(paths)}
        )

        preds: List[np.ndarray] = []
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Prediction file missing: {p}")
            preds.append(np.load(p))

        # -----------------------------
        # Shape safety
        # -----------------------------
        base_shape = preds[0].shape
        for i, arr in enumerate(preds):
            if arr.shape != base_shape:
                raise ValueError(
                    f"Prediction shape mismatch at index {i}: "
                    f"{arr.shape} vs {base_shape}"
                )

        self.monitor.log_stage_end("load_predictions", {"status": "success"})
        return preds

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        method = self.ensemble_cfg.get("method", "mean").lower()
        metrics = self.ensemble_cfg.get("metrics", ["rmse", "mae"])

        logger.info("[ENS] Running Pipeline F | method=%s", method)
        self.monitor.log_stage_start("F_ensemble_pipeline", {"method": method})

        # ==============================================================
        # Simple Ensembles
        # ==============================================================
        if method in {"mean", "median", "weighted"}:
            pred_paths = [
                Path(p) for p in self.ensemble_cfg.get("pred_paths", [])
            ]
            if not pred_paths:
                raise ValueError("ensemble.pred_paths must be provided")

            y_true_path = Path(self.ensemble_cfg["y_true_path"])
            if not y_true_path.exists():
                raise FileNotFoundError(f"y_true not found: {y_true_path}")

            preds = self._load_predictions(pred_paths)
            ensembler = SimpleEnsembler(predictions=preds)

            self.monitor.log_stage_start("ensemble_prediction_generation")

            if method == "weighted":
                weights = self.ensemble_cfg.get("weights")
                if not weights:
                    raise ValueError("weights must be provided for weighted ensemble")

                if len(weights) != len(preds):
                    raise ValueError(
                        "weights length must match number of models"
                    )

                y_pred = ensembler.ensemble_predictions(
                    method="weighted",
                    weights=weights,
                )
            else:
                y_pred = ensembler.ensemble_predictions(method)

            self.monitor.log_stage_end(
                "ensemble_prediction_generation", {"status": "success"}
            )

            # -----------------------------
            # Save predictions
            # -----------------------------
            pred_out_path = self.pred_dir / f"{method}_ensemble.npy"
            np.save(pred_out_path, y_pred)

            # -----------------------------
            # Evaluation
            # -----------------------------
            y_true = np.load(y_true_path)
            results = compute_metrics(y_true, y_pred, metrics)

            # -----------------------------
            # Persist summary
            # -----------------------------
            summary = {
                "run_id": self.run_id,
                "method": method,
                "models": [p.name for p in pred_paths],
                "metrics": results,
            }

            (self.ensemble_dir / "ensemble_summary.json").write_text(
                json.dumps(summary, indent=2)
            )

            logger.info("[ENS] Pipeline F completed | results=%s", results)
            self.monitor.log_stage_end(
                "F_ensemble_pipeline", {"status": "completed"}
            )
            return results

        # ==============================================================
        # Stacked Ensemble
        # ==============================================================
        elif method == "stacked":
            self.monitor.log_stage_start("stacked_ensemble")

            oof_cfg = self.ensemble_cfg.get("oof", {})
            X_path = Path(oof_cfg["X_path"])
            y_path = Path(oof_cfg["y_path"])
            n_splits = int(oof_cfg.get("n_splits", 5))
            model_params = oof_cfg.get("model_params", {})

            gen = OOFGenerator(
                model_params=model_params,
                n_splits=n_splits,
            )

            X, y = gen.load_data(X_path, y_path)
            oof_preds, oof_targets = gen.generate(X, y)

            oof_preds_path = self.oof_dir / "oof_preds.npy"
            oof_targets_path = self.oof_dir / "oof_targets.npy"

            np.save(oof_preds_path, oof_preds)
            np.save(oof_targets_path, oof_targets)

            # -----------------------------
            # Meta-features
            # -----------------------------
            mf_cfg = self.ensemble_cfg.get("meta_features", {})
            meta_csv_path = self.meta_dir / "meta_features.csv"

            mf_builder = MetaFeaturesBuilder(
                oof_preds_path=oof_preds_path,
                oof_targets_path=oof_targets_path,
                feature_paths=mf_cfg.get("feature_paths", []),
            )

            self.monitor.log_stage_start("meta_feature_construction")
            mf_builder.build(save_path=meta_csv_path)
            self.monitor.log_stage_end(
                "meta_feature_construction", {"status": "success"}
            )

            # -----------------------------
            # Meta-model
            # -----------------------------
            mm_cfg = self.ensemble_cfg.get("meta_model", {})
            model_save_path = self.meta_dir / "meta_model.pkl"

            trainer = MetaFeatureTrainer(
                data_path=meta_csv_path,
                test_size=float(mm_cfg.get("test_size", 0.2)),
                random_state=int(mm_cfg.get("random_state", 42)),
                model_params=mm_cfg.get("params", {}),
            )

            self.monitor.log_stage_start("meta_model_training")
            X_train, y_train, X_val, y_val = trainer.prepare_datasets()
            trainer.train_model(X_train, y_train, X_val, y_val)
            results = trainer.evaluate(X_val, y_val)
            trainer.save_model(model_save_path)
            self.monitor.log_stage_end(
                "meta_model_training", {"status": "success"}
            )

            logger.info("Pipeline F (stacked) completed | results=%s", results)
            self.monitor.log_stage_end(
                "F_ensemble_pipeline", {"status": "completed"}
            )
            return results

        # ==============================================================
        # Unsupported
        # ==============================================================
        else:
            raise ValueError(f"Unsupported ensemble method: {method}")
