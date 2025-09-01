"""
Module to orchestrate walk-forward validation for equity forecasting models.

This module loads configuration, runs walk-forward validation using the
WalkForwardValidator class, and logs summarized results.

It is intended to be called by the single CLI entrypoint (e.g., src/main.py).
"""

from typing import Dict, Any
from src.training.walk_forward_validator import WalkForwardValidator
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

def run_walk_forward_validation(config_path: str) -> Dict[str, Any]:
    """
    Perform walk-forward validation based on provided config file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        Dict[str, Any]: Aggregated results of the validation (metrics, details)
    """
    # Load the config file
    config = load_config(config_path)

    # Extract early stopping config if available
    early_stopping_cfg = config.get("early_stopping", None)

    # Initialize the walk-forward validator with early stopping support
    wfv = WalkForwardValidator(config=config, early_stopping=early_stopping_cfg)

    # Run the validation, which returns detailed results
    results = wfv.run_validation()

    # Log summarized results (mean metrics across all splits)
    for metric, value in results["summary"].items():
        logger.info(f"Walk-forward validation metric - {metric}: {value:.4f}")

    return results
