"""
pipeline_wrapper.py

Thin wrapper functions to expose pipeline stages for Orchestrator and CLI.
These wrappers delegate execution to the appropriate pipeline classes.
"""

from src.pipeline.run_training import ModelTrainerPipeline as run_training
from src.pipeline.run_optimizer import run_hyperparameter_optimization as run_optimizer
from src.pipeline.run_ensemble import run_ensemble
from src.pipeline.run_prediction import run_prediction_pipeline as run_prediction
from src.pipeline.run_walk_forward import run_walk_forward_validation as run_walk_forward


def run_training(config_path: str) -> None:
    """Wrapper for training stage."""
    run_training(config_path).run()


def run_optimizer(config_path: str) -> None:
    """Wrapper for hyperparameter optimization stage."""
    run_optimizer(config_path).run()


def run_ensemble(config_path: str) -> None:
    """Wrapper for ensembling stage."""
    run_ensemble(config_path).run()


def run_prediction(config_path: str) -> None:
    """Wrapper for prediction stage."""
    run_prediction(config_path).run()


def run_walk_forward(config_path: str) -> None:
    """Wrapper for walk-forward validation stage."""
    run_walk_forward(config_path).run()
