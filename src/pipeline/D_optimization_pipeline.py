# src/pipeline/D_optimization_pipeline.py

"""
Module to orchestrate hyperparameter optimization for equity forecasting models.

This module loads configuration, training data, selects the optimizer backend dynamically,
runs the hyperparameter search, and logs the progress.

⚠️ IMPORTANT WARNING:
Do NOT call these functions directly in end-user workflows.
Use wrappers in src/pipeline/pipeline_wrapper.py to enforce orchestration, logging,
retries, and task markers.
"""

from typing import Dict, Tuple
from datetime import datetime, timezone
import numpy as np

from src.utils.logger import get_logger
from src.optimizers import get_optimizer
from src.utils.config import load_config
from src.monitoring.monitor import TrainingMonitor

logger = get_logger(__name__)


def load_training_data(
    x_path: str,
    y_path: str,
    monitor: TrainingMonitor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training features and targets from given file paths.
    """
    monitor.log_stage_start("load_training_data", {"X_path": x_path, "y_path": y_path})
    try:
        X = np.load(x_path)
        y = np.load(y_path)
        logger.info("Loaded training data: X=%s y=%s", X.shape, y.shape)
        monitor.log_stage_end("load_training_data", {"status": "success"})
        return X, y
    except Exception as e:
        monitor.log_stage_end(
            "load_training_data",
            {"status": "failed", "error": str(e)},
        )
        raise


def run_hyperparameter_optimization(
    config_path: str,
    optimizer_name: str = "optuna",
) -> None:
    """
    Main orchestration function for running hyperparameter optimization.
    """
    if not config_path:
        raise ValueError("config_path must be provided")

    # -------------------------------------------------
    # Load config
    # -------------------------------------------------
    config: Dict = load_config(config_path)
    train_cfg = config.get("training", {})
    optim_cfg = config.get("optimization", {})

    # -------------------------------------------------
    # Run identity
    # -------------------------------------------------
    scope = train_cfg.get("scope", "GLOBAL")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{scope}_OPTIM_{timestamp}"

    base_run_dir = optim_cfg.get("run_dir", "runs/optimization")
    run_dir = f"{base_run_dir}/{run_id}"

    # -------------------------------------------------
    # Runtime monitor (CORRECT)
    # -------------------------------------------------
    monitor = TrainingMonitor(
        run_id=run_id,
        save_dir=run_dir,
        visualize=False,
        flush_every=int(optim_cfg.get("flush_every", 1)),
    )

    monitor.log_stage_start(
        "run_hyperparameter_optimization",
        {"config_path": config_path, "optimizer": optimizer_name},
    )

    try:
        # -------------------------------------------------
        # Load data
        # -------------------------------------------------
        x_path = config["data"]["X_train_path"]
        y_path = config["data"]["y_train_path"]
        X_train, y_train = load_training_data(x_path, y_path, monitor)

        # -------------------------------------------------
        # Optimizer
        # -------------------------------------------------
        optimizer_func = get_optimizer(optimizer_name)
        n_trials = train_cfg.get("n_trials", 50)

        monitor.log_stage_start(
            "hyperparameter_search",
            {"n_trials": n_trials, "optimizer": optimizer_name},
        )

        optimizer_func(
            config=config,
            X=X_train,
            y=y_train,
            n_trials=n_trials,
        )

        monitor.log_stage_end("hyperparameter_search", {"status": "success"})
        monitor.log_stage_end("run_hyperparameter_optimization", {"status": "completed"})

    except Exception as e:
        monitor.log_stage_end(
            "run_hyperparameter_optimization",
            {"status": "failed", "error": str(e)},
        )
        raise
