# src/pipeline/run_walk_forward.py

"""
Module to orchestrate walk-forward validation for equity forecasting models.

This module loads configuration, runs walk-forward validation using the
WalkForwardValidator class, and logs summarized results.

⚠️ IMPORTANT WARNING:
Do NOT call these functions/classes directly. Use the wrapper functions
in src/pipeline/pipeline_wrapper.py to enforce orchestration, logging,
retries, and task markers.
"""

from typing import Dict, Any
from src.training.walk_forward_validator import WalkForwardValidator
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.monitoring.monitor import TrainingMonitor

logger = get_logger(__name__)
monitor = TrainingMonitor()


def run_walk_forward_validation(config_path: str) -> Dict[str, Any]:
    """
    Perform walk-forward validation based on provided config file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        Dict[str, Any]: Aggregated results of the validation (metrics, details)
    """
    monitor.log_stage_start("run_walk_forward_validation", {"config_path": config_path})
    logger.info(f"Starting walk-forward validation using config: {config_path}")

    # Load the config file
    config = load_config(config_path)
    monitor.log_stage_start("load_config", {"config_path": config_path})
    monitor.log_stage_end("load_config", {"status": "success"})

    # Extract early stopping config if available
    early_stopping_cfg = config.get("early_stopping", None)

    # Initialize the walk-forward validator with early stopping support
    monitor.log_stage_start("initialize_validator")
    wfv = WalkForwardValidator(config=config, early_stopping=early_stopping_cfg)
    monitor.log_stage_end("initialize_validator", {"status": "success"})

    # Run the validation, which returns detailed results
    monitor.log_stage_start("run_validation")
    results = wfv.run_validation()
    monitor.log_stage_end("run_validation", {"status": "success"})

    # Log summarized results (mean metrics across all splits)
    monitor.log_stage_start("log_results_summary")
    for metric, value in results.get("summary", {}).items():
        logger.info(f"Walk-forward validation metric - {metric}: {value:.4f}")
    monitor.log_stage_end("log_results_summary", {"num_metrics": len(results.get("summary", {}))})

    monitor.log_stage_end("run_walk_forward_validation", {"status": "completed"})
    return results
