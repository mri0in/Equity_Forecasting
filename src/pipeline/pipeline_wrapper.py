# src/pipeline/pipeline_wrapper.py

"""
Pipeline wrapper functions with monitoring integration.

These wrappers expose the pipeline stages to the orchestrator or CLI.
Each stage now logs start/end events and captures metrics or errors.
"""

from typing import Optional
from src.pipeline.run_ingestion import IngestionPipeline
from src.pipeline.run_training import ModelTrainerPipeline
from src.pipeline.run_optimizer import run_hyperparameter_optimization
from src.pipeline.run_ensemble import run_ensemble, load_ensemble_config
from src.pipeline.run_prediction import run_prediction_pipeline
from src.pipeline.run_walk_forward import run_walk_forward_validation
from src.monitoring.monitor import log_stage_start, log_stage_end
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_ingestion(tickers: list, start_date: str = "2013-01-01", end_date: str = None) -> None:
    """
    Wrapper for ingestion stage with monitoring/logging.
    """
    stage_name = "ingestion_pipeline"
    log_stage_start(stage_name, details={"tickers": tickers})

    try:
        pipeline = IngestionPipeline(tickers, start_date, end_date)
        pipeline.run()

        log_stage_end(stage_name, metrics={"status": "completed"})
    except Exception as e:
        log_stage_end(stage_name, metrics={"status": "failed", "error": str(e)})
        raise

def run_training(config_path: str) -> None:
    """Wrapper for training stage with monitoring."""
    stage_name = "training_pipeline"
    log_stage_start(stage_name, details={"config_path": config_path})
    try:
        pipeline = ModelTrainerPipeline(config_path)
        pipeline.run()
        log_stage_end(stage_name, metrics={"status": "completed"})
    except Exception as e:
        log_stage_end(stage_name, metrics={"status": "failed", "error": str(e)})
        raise


def run_optimizer(config_path: str, optimizer_name: str = "optuna") -> None:
    """Wrapper for hyperparameter optimization stage with monitoring."""
    stage_name = "hyperparameter_optimization"
    log_stage_start(stage_name, details={"config_path": config_path, "optimizer": optimizer_name})
    try:
        run_hyperparameter_optimization(config_path, optimizer_name)
        log_stage_end(stage_name, metrics={"status": "completed"})
    except Exception as e:
        log_stage_end(stage_name, metrics={"status": "failed", "error": str(e)})
        raise


def run_ensemble(config_path: str):
    """Wrapper for ensemble stage with monitoring."""
    stage_name = "ensemble_pipeline"
    log_stage_start(stage_name, details={"config_path": config_path})
    try:
        config = load_ensemble_config(config_path)
        results = run_ensemble(config)
        log_stage_end(stage_name, metrics={"status": "completed", "metrics": results})
        return results
    except Exception as e:
        log_stage_end(stage_name, metrics={"status": "failed", "error": str(e)})
        raise


def run_prediction(config_path: str, ticker: Optional[str] = None) -> None:
    """Wrapper for prediction stage with monitoring."""
    stage_name = "prediction_pipeline"
    log_stage_start(stage_name, details={"config_path": config_path, "ticker": ticker})
    try:
        run_prediction_pipeline(config_path, ticker)
        log_stage_end(stage_name, metrics={"status": "completed"})
    except Exception as e:
        log_stage_end(stage_name, metrics={"status": "failed", "error": str(e)})
        raise


def run_walk_forward(config_path: str):
    """Wrapper for walk-forward validation stage with monitoring."""
    stage_name = "walk_forward_validation"
    log_stage_start(stage_name, details={"config_path": config_path})
    try:
        results = run_walk_forward_validation(config_path)
        log_stage_end(stage_name, metrics={"status": "completed", "summary": results.get("summary", {})})
        return results
    except Exception as e:
        log_stage_end(stage_name, metrics={"status": "failed", "error": str(e)})
        raise
