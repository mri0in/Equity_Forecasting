# src/pipeline/E_ensemble_pipeline.py """ Ensemble Pipeline (E).
"""
- Consumes predictions from base models (D or F)
- Applies mean / weighted / median / stacked ensemble
- Evaluates ensemble predictions
- Saves all artifacts under runtime `run_id`:
  run_id/
    └─ ensemble/
        ├─ oof/
        ├─ meta/
        └─ predictions/ 
"""
import os
from datetime import datetime, timezone
from typing import Dict, List 
import numpy as np
import yaml

from src.utils.logger import get_logger
from src.monitoring.monitor import TrainingMonitor
from src.ensemble.simple_ensembler import SimpleEnsembler
from src.ensemble.generate_oof import OOFGenerator
from src.ensemble.meta_features import MetaFeaturesBuilder
from src.ensemble.train_meta_features import MetaFeatureTrainer
from src.training.evaluate import compute_metrics

logger = get_logger(__name__)

# ------------------------------
# Helper to load predictions
# ------------------------------
def _load_pred_arrays(paths: List[str], monitor: TrainingMonitor) -> List[np.ndarray]:
    monitor.log_stage_start("load_pred_arrays", {"num_paths": len(paths)})
    try:
        preds = [np.load(p) for p in paths]
        logger.info("Loaded %d prediction arrays", len(preds))
        monitor.log_stage_end("load_pred_arrays", {"status": "success"})
        return preds
    except Exception as e:
        monitor.log_stage_end("load_pred_arrays", {"status": "failed", "error": str(e)})
        raise

# ------------------------------
# Main Ensemble Runner
# ------------------------------
def run_ensemble(config_path: str) -> Dict[str, float]:
    if not config_path:
        raise ValueError("config_path must be provided")

    with open(config_path, "r") as f:
        config: Dict = yaml.safe_load(f)

    ensemble_cfg = config.get("ensemble", {})
    train_cfg = config.get("training", {})

    # ------------------------------
    # runtime identifiers
    # ------------------------------
    scope = train_cfg.get("scope", "GLOBAL")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{scope}_ENSEMBLE_{timestamp}"
    run_dir = os.path.join(ensemble_cfg.get("run_dir", "runs/ensemble"), run_id)
    os.makedirs(run_dir, exist_ok=True)

    monitor = TrainingMonitor(
        run_id=run_id,
        save_dir=run_dir,
        visualize=False,
        flush_every=int(ensemble_cfg.get("flush_every", 1)),
    )

    method = ensemble_cfg.get("method", "mean").lower()
    logger.info("Running ensemble method: %s", method)
    monitor.log_stage_start("run_ensemble", {"method": method, "config_path": config_path})

    # ===============================
    # Simple / Weighted / Median
    # ===============================
    if method in {"mean", "weighted", "median"}:
        pred_paths: List[str] = ensemble_cfg.get("pred_paths", [])
        if not pred_paths:
            raise ValueError("ensemble.pred_paths must be provided")

        y_true_path: str = ensemble_cfg["y_true_path"]
        metrics: List[str] = ensemble_cfg.get("metrics", ["rmse", "mae"])

        preds = _load_pred_arrays(pred_paths, monitor)
        ensembler = SimpleEnsembler(predictions=preds)

        monitor.log_stage_start("ensemble_predictions", {"method": method})
        if method == "weighted":
            weights: List[float] = ensemble_cfg["weights"]
            y_pred = ensembler.ensemble_predictions("weighted", weights=weights)
        elif method == "median":
            y_pred = ensembler.ensemble_predictions("median")
        else:
            y_pred = ensembler.ensemble_predictions("mean")
        monitor.log_stage_end("ensemble_predictions", {"status": "success"})

        # ------------------------------
        # Save ensemble predictions
        # ------------------------------
        pred_dir = os.path.join(run_dir, "ensemble", "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        pred_path = os.path.join(pred_dir, f"{method}_ensemble.npy")
        np.save(pred_path, y_pred)
        logger.info("Saved ensemble predictions to %s", pred_path)

        y_true = np.load(y_true_path)
        results = compute_metrics(y_true, y_pred, metrics)

        logger.info("%s ensemble evaluation: %s", method.capitalize(), results)
        monitor.log_stage_end("run_ensemble", {"status": "completed"})
        return results

    # ===============================
    # Stacked Ensemble (OOF → Meta → Meta-model)
    # ===============================
    elif method == "stacked":
        monitor.log_stage_start("stacked_ensemble_pipeline")

        oof_cfg = ensemble_cfg.get("oof", {})
        X_path = oof_cfg["X_path"]
        y_path = oof_cfg["y_path"]
        n_splits = int(oof_cfg.get("n_splits", 5))
        model_params = oof_cfg.get("model_params", {})

        oof_out_dir = os.path.join(run_dir, "ensemble", "oof")
        os.makedirs(oof_out_dir, exist_ok=True)

        gen = OOFGenerator(model_params=model_params, n_splits=n_splits)
        X, y = gen.load_data(X_path, y_path)
        oof_preds, oof_targets = gen.generate(X, y)

        oof_preds_path = os.path.join(oof_out_dir, "oof_preds.npy")
        oof_targets_path = os.path.join(oof_out_dir, "oof_targets.npy")
        np.save(oof_preds_path, oof_preds)
        np.save(oof_targets_path, oof_targets)
        logger.info("Saved OOF arrays to %s", oof_out_dir)

        # ------------------------------
        # Meta-features
        # ------------------------------
        mf_cfg = ensemble_cfg.get("meta_features", {})
        meta_csv_path = os.path.join(run_dir, "ensemble", "meta", "meta_features.csv")
        os.makedirs(os.path.dirname(meta_csv_path), exist_ok=True)

        mf_builder = MetaFeaturesBuilder(
            oof_preds_path=oof_preds_path,
            oof_targets_path=oof_targets_path,
            feature_paths=mf_cfg.get("feature_paths", []),
        )
        monitor.log_stage_start("build_meta_features")
        mf_builder.build(save_path=meta_csv_path)
        monitor.log_stage_end("build_meta_features", {"status": "success"})

        # ------------------------------
        # Meta-model
        # ------------------------------
        mm_cfg = ensemble_cfg.get("meta_model", {})
        model_save_path = os.path.join(run_dir, "ensemble", "meta", "meta_model.pkl")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        trainer = MetaFeatureTrainer(
            data_path=meta_csv_path,
            test_size=float(mm_cfg.get("test_size", 0.2)),
            random_state=int(mm_cfg.get("random_state", 42)),
            model_params=mm_cfg.get("params"),
        )

        monitor.log_stage_start("train_meta_model")
        X_train, y_train, X_val, y_val = trainer.prepare_datasets()
        trainer.train_model(X_train, y_train, X_val, y_val)
        results = trainer.evaluate(X_val, y_val)
        trainer.save_model(model_save_path)
        monitor.log_stage_end("train_meta_model", {"status": "success"})

        logger.info("Stacked ensemble evaluation: %s", results)
        monitor.log_stage_end("stacked_ensemble_pipeline", {"status": "completed"})
        monitor.log_stage_end("run_ensemble", {"status": "completed"})
        return results

    else:
        monitor.log_stage_end("run_ensemble", {"status": "failed", "error": f"Unsupported method: {method}"})
        raise ValueError(f"Unsupported ensemble method: {method}")
