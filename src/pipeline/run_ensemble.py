"""
This module handles the orchestration of ensemble modeling.
It loads base model predictions, applies an ensemble strategy,
evaluates the final ensemble prediction, and logs the results.

⚠️ IMPORTANT WARNING FOR USERS & DEVELOPERS
# For orchestration and end-user workflows, DO NOT call these classes
# directly. Instead, always use the wrapper functions in:
#
#     src/pipeline/pipeline_wrapper.py
#
# Example:
#     from src.pipeline.pipeline_wrapper import run_ensemble
#     run_ensemble("configs/ensemble_config.yaml")
#
# Reason:
# The wrappers provide a consistent interface for the orchestrator and enforce
# config-driven execution across the project. Direct class calls may bypass
# orchestration safeguards (retries, logging, markers).
# -------
"""

# src/pipeline/run_ensemble.py

import os
from typing import Dict, List

import numpy as np
import yaml

from src.utils.logger import get_logger
from src.ensemble.simple_ensembler import SimpleEnsembler
from src.ensemble.generate_oof import OOFGenerator
from src.ensemble.meta_features import MetaFeaturesBuilder
from src.ensemble.train_meta_features import MetaFeatureTrainer
from src.training.evaluate import compute_metrics

logger = get_logger(__name__)


def load_ensemble_config(config_path: str) -> Dict:
    """Load ensemble configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded ensemble config from %s", config_path)
    return config


def _load_pred_arrays(paths: List[str]) -> List[np.ndarray]:
    preds = [np.load(p) for p in paths]
    logger.info("Loaded %d prediction arrays", len(preds))
    return preds


def run_ensemble(config: Dict) -> Dict[str, float]:
    """
    Run the selected ensemble strategy based on config.
    Returns a dict of evaluation metrics.
    """
    ensemble_cfg = config.get("ensemble", {})
    method = ensemble_cfg.get("method", "mean").lower()
    logger.info("Running ensemble method: %s", method)

    # ---------- Simple / Weighted / Median ----------
    if method in {"mean", "weighted", "median"}:
        pred_paths: List[str] = ensemble_cfg.get("pred_paths", [])
        if not pred_paths:
            raise ValueError("No prediction paths provided for mean ensemble. Please specify 'pred_paths' in config.")

        y_true_path: str = ensemble_cfg["y_true_path"]
        metrics: List[str] = ensemble_cfg.get("metrics", ["rmse", "mae"])

        preds = _load_pred_arrays(pred_paths)
        ensembler = SimpleEnsembler(predictions=preds)

        if method == "weighted":
            weights: List[float] = ensemble_cfg["weights"]
            y_pred = ensembler.ensemble_predictions(method="weighted", weights=weights)
        elif method == "median":
            y_pred = ensembler.ensemble_predictions(method="median")
        else:  # mean
            y_pred = ensembler.ensemble_predictions(method="mean")

        y_true = np.load(y_true_path)
        results = compute_metrics(y_true, y_pred, metrics)
        logger.info("%s ensemble evaluation: %s", method.capitalize(), results)
        return results

    # ---------- Stacked (OOF -> Meta Features -> Meta Model) ----------
    elif method == "stacked":
        oof_cfg = ensemble_cfg.get("oof", {})
        X_path: str = oof_cfg["X_path"]
        y_path: str = oof_cfg["y_path"]
        n_splits: int = oof_cfg.get("n_splits", 5)
        model_params: Dict = oof_cfg.get("model_params", {})
        oof_out_dir: str = oof_cfg.get("out_dir", "datalake/ensemble/oof")

        # Generate OOF predictions
        gen = OOFGenerator(model_params=model_params, n_splits=n_splits)
        X, y = gen.load_data(X_path, y_path)
        oof_preds, oof_targets = gen.generate(X, y)

        # Persist OOF arrays for MetaFeaturesBuilder (expects file paths)
        os.makedirs(oof_out_dir, exist_ok=True)
        oof_preds_path = os.path.join(oof_out_dir, "oof_preds.npy")
        oof_targets_path = os.path.join(oof_out_dir, "oof_targets.npy")
        np.save(oof_preds_path, oof_preds)
        np.save(oof_targets_path, oof_targets)
        logger.info("Saved OOF arrays to %s", oof_out_dir)

        # Build meta-features
        mf_cfg = ensemble_cfg.get("meta_features", {})
        feature_paths: List[str] = mf_cfg.get("feature_paths", [])
        meta_csv_path: str = mf_cfg.get("save_path", "datalake/ensemble/meta/meta_features.csv")

        mf_builder = MetaFeaturesBuilder(
            oof_preds_path=oof_preds_path,
            oof_targets_path=oof_targets_path,
            feature_paths=feature_paths,
        )
        _ = mf_builder.build(save_path=meta_csv_path)

        # Train/evaluate meta model
        mm_cfg = ensemble_cfg.get("meta_model", {})
        model_save_path: str = mm_cfg.get("save_path", "datalake/ensemble/meta/meta_model.pkl")
        test_size: float = mm_cfg.get("test_size", 0.2)
        random_state: int = mm_cfg.get("random_state", 42)
        model_hparams: Dict = mm_cfg.get("params", None)  # optional dict for LightGBM

        trainer = MetaFeatureTrainer(
            data_path=meta_csv_path,
            test_size=test_size,
            random_state=random_state,
            model_params=model_hparams,
        )

        X_train, y_train, X_val, y_val = trainer.prepare_datasets()
        trainer.train_model(X_train, y_train, X_val, y_val)
        results = trainer.evaluate(X_val, y_val)
        trainer.save_model(model_save_path)

        logger.info("Stacked ensemble evaluation: %s", results)
        return results

    else:
        logger.error("Unsupported ensemble method: %s", method)
        raise ValueError(f"Unsupported ensemble method: {method}")
