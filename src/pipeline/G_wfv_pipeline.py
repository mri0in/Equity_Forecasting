# src/pipeline/G_wfv_pipeline.py

"""
Module to orchestrate walk-forward validation for equity forecasting models.

This module loads configuration, runs walk-forward validation using the
WalkForwardValidator class, and logs summarized results.

⚠️ IMPORTANT WARNING:
Do NOT call these functions/classes directly.
Use wrapper functions in src/pipeline/pipeline_wrapper.py
to enforce orchestration, logging, retries, and task markers.
"""

from datetime import datetime, timezone
from typing import Dict, Any

from src.training.walk_forward_validator import WalkForwardValidator
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.monitoring.monitor import TrainingMonitor

logger = get_logger(__name__)


def run_walk_forward_validation(config_path: str) -> Dict[str, Any]:
    """
    Perform walk-forward validation based on provided config file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        Dict[str, Any]: Aggregated validation results
    """
    if not config_path:
        raise ValueError("config_path must be provided")

    # -------------------------------------------------
    # Load config
    # -------------------------------------------------
    config = load_config(config_path)
    train_cfg = config.get("training", {})

    # -------------------------------------------------
    # Run identity
    # -------------------------------------------------
    scope = train_cfg.get("scope", "GLOBAL")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{scope}_WALKFWD_{timestamp}"

    base_run_dir = train_cfg.get("run_dir", "runs/walk_forward")
    run_dir = f"{base_run_dir}/{run_id}"

    # -------------------------------------------------
    # Runtime monitor (CORRECT)
    # -------------------------------------------------
    monitor = TrainingMonitor(
        run_id=run_id,
        save_dir=run_dir,
        visualize=False,
        flush_every=int(train_cfg.get("flush_every", 1)),
    )

    monitor.log_stage_start(
        "run_walk_forward_validation",
        {"config_path": config_path, "run_id": run_id},
    )

    logger.info(
        "Starting walk-forward validation | run_id=%s | config=%s",
        run_id,
        config_path,
    )

    # -------------------------------------------------
    # Extract early stopping config
    # -------------------------------------------------
    early_stopping_cfg = config.get("early_stopping")

    # -------------------------------------------------
    # Initialize validator
    # -------------------------------------------------
    monitor.log_stage_start("initialize_validator")
    wfv = WalkForwardValidator(
        config=config,
        early_stopping=early_stopping_cfg,
    )
    monitor.log_stage_end("initialize_validator", {"status": "success"})

    # -------------------------------------------------
    # Run validation
    # -------------------------------------------------
    monitor.log_stage_start("run_validation")
    results = wfv.run_validation()
    monitor.log_stage_end("run_validation", {"status": "success"})

    # -------------------------------------------------
    # Log summarized metrics
    # -------------------------------------------------
    summary = results.get("summary", {})
    monitor.log_stage_start(
        "log_results_summary",
        {"num_metrics": len(summary)},
    )

    for metric, value in summary.items():
        logger.info(
            "Walk-forward metric | %s = %.4f",
            metric,
            value,
        )

    monitor.log_stage_end("log_results_summary", {"status": "success"})

    monitor.log_stage_end(
        "run_walk_forward_validation",
        {"status": "completed"},
    )

    return results
