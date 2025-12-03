"""
Router for optimizers module.
"""

from typing import Callable, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Define a mapping of optimizer names to their runner functions
OPTIMIZERS: Dict[str, Callable] = {}

try:
    from .optuna_optimizer import run_optimization as run_optuna
    OPTIMIZERS["optuna"] = run_optuna
except ImportError as e:
    logger.warning("Optuna optimizer module not found: %s", e)


def get_optimizer(name: str) -> Callable:
    """
    Retrieve the optimizer runner function by name.

    Args:
        name (str): Optimizer name (e.g. 'optuna')

    Returns:
        Callable: The optimization runner function

    Raises:
        ValueError: If optimizer not found
    """
    optimizer = OPTIMIZERS.get(name.lower())
    if optimizer is None:
        valid_opts = list(OPTIMIZERS.keys())
        raise ValueError(f"Optimizer '{name}' not supported. Available: {valid_opts}")
    return optimizer
