# src/dag/stages.py
"""
PipelineStages Wrappers for DAG Execution
-----------------------------------------

Each method here corresponds to a DAG node and wraps
existing pipeline/ensemble functions from src/pipeline or src/ensemble.

"""

import logging
from typing import Any

from src.pipeline import (
    run_training,
    run_optimizer,
    run_walk_forward,
    run_ensemble,
    run_prediction,
    pipeline_wrapper
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PipelineStages:
    """
    Wrapper class exposing all pipeline stages as simple callable methods.

    Each method should execute a single DAG stage and raise exceptions
    on failure (handled by DAGRunner retries).
    """

    def run_ingestion(self) -> None:
        """
        Data ingestion stage: load from API/CSV or other sources.
        """
        logger.info("Starting ingestion stage...")
        pipeline_wrapper.run_ingestion()  # Call your pipeline wrapper logic
        logger.info("Ingestion stage completed.")

    def run_preprocessing(self) -> None:
        """
        Preprocessing stage: cleaning, imputing, scaling, etc.
        """
        logger.info("Starting preprocessing stage...")
        pipeline_wrapper.run_preprocessing()
        logger.info("Preprocessing stage completed.")

    def run_feature_generation(self) -> None:
        """
        Feature generation stage: technical + cached features.
        """
        logger.info("Starting feature generation stage...")
        pipeline_wrapper.run_feature_generation()
        logger.info("Feature generation stage completed.")

    def run_training(self) -> None:
        """
        Model training stage.
        """
        logger.info("Starting training stage...")
        run_training.run_training_pipeline()
        logger.info("Training stage completed.")

    def run_optimizer(self) -> None:
        """
        Hyperparameter optimization stage.
        """
        logger.info("Starting optimizer stage...")
        run_optimizer.run_optimizer_pipeline()
        logger.info("Optimizer stage completed.")

    def run_walkforward(self) -> None:
        """
        Walk-forward validation stage.
        """
        logger.info("Starting walk-forward validation stage...")
        run_walk_forward.run_walk_forward_pipeline()
        logger.info("Walk-forward validation stage completed.")

    def run_ensembling(self) -> None:
        """
        Ensembling stage: generate meta-features, train meta-models, simple ensemble.
        """
        logger.info("Starting ensembling stage...")
        run_ensemble.run_ensemble_pipeline()
        logger.info("Ensembling stage completed.")

    def run_forecasting(self) -> None:
        """
        Forecasting stage: generate predictions for a given equity.
        """
        logger.info("Starting forecasting stage...")
        run_prediction.run_prediction_pipeline()
        logger.info("Forecasting stage completed.")
