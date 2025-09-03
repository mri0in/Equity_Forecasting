"""
orchestrator.py

Master orchestrator for the Equity Forecasting project.
Coordinates all pipeline stages: training, optimization, validation,
ensembling, and prediction. Implements Wall Street-grade orchestration.
"""

import logging
import time
import os
from typing import List

from src.pipeline import (
    run_training,
    run_optimizer,
    run_walk_forward,
    run_ensemble,
    run_prediction,
)

from src.utils.config_loader import load_typed_config, FullConfig

# -------------------------------
# Completion markers for each pipeline task
# -------------------------------
TASK_MARKERS = {
    "train": "datalake/models/trained/.train_complete",
    "optimize": "datalake/experiments/optuna/.optimize_complete",
    "ensemble": "datalake/ensemble/.ensemble_complete",
    "predict": "datalake/predictions/.predict_complete",
    "walkforward": "datalake/wfv/.walkforward_complete",
}

class PipelineOrchestrator:
    """
    Orchestrates the full equity forecasting pipeline based on config.

    Attributes:
        config_path (str): Path to YAML config file.
        config (FullConfig): Parsed configuration object.
        pipeline_cfg: Pipeline section of config.
        logger: Logger instance.
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config: FullConfig = load_typed_config(config_path)
        self.pipeline_cfg = self.config.pipeline
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_task(self, task: str, retries: int = 1, strict: bool = True) -> None:
        """
        Run a single pipeline task with retries and strictness handling.

        Args:
            task (str): Task name (train, optimize, ensemble, predict, walkforward).
            retries (int): Number of times to retry if the task fails.
            strict (bool): If True, pipeline stops on failure. If False, continues.

        Raises:
            Exception: Re-raises last error if strict is True and all retries fail.
        """
        marker = TASK_MARKERS.get(task)
        if marker and os.path.exists(marker):
            self.logger.info(f"Skipping task '{task}' — completion marker exists: {marker}")
            return

        attempt = 0
        while attempt < retries:
            try:
                self.logger.info(f"Running task: {task} (attempt {attempt + 1}/{retries})")

                if task == "train":
                    run_training(self.config_path)
                elif task == "optimize":
                    run_optimizer(self.config_path)
                elif task == "ensemble":
                    run_ensemble(self.config_path)
                elif task == "predict":
                    run_prediction(self.config_path)
                elif task == "walkforward":
                    run_walk_forward(self.config_path)
                else:
                    self.logger.warning(f"Unknown task '{task}' — skipping")
                return  # success → exit

            except Exception as e:
                attempt += 1
                self.logger.error(f"Task '{task}' failed on attempt {attempt}/{retries}: {e}")
                if attempt >= retries:
                    if strict:
                        self.logger.critical(f"Task '{task}' failed after {retries} attempts. Aborting.")
                        raise
                    else:
                        self.logger.warning(f"Skipping failed task '{task}' (strict={strict})")
                        return

    def run_pipeline(self, tasks: list = None) -> None:
        """
        Run the full pipeline as defined in the config or a custom task list.

        Args:
            tasks (list, optional): List of tasks to run. 
                                    If None, uses the config-defined order.
        """
        # Use config-defined tasks if no custom list is provided
        tasks_to_run = tasks if tasks else self.pipeline_cfg.tasks
        retries = self.pipeline_cfg.retries
        strict = self.pipeline_cfg.strict

        self.logger.info(f"Starting orchestrated pipeline with tasks: {tasks_to_run}")
        for task in tasks_to_run:
            self.run_task(task, retries=retries, strict=strict)
        self.logger.info("Pipeline execution completed successfully.")

    def reset_task_markers(self, tasks: list = None) -> None:
        """
        Remove completion markers to force tasks to rerun.

        Args:
            tasks (list, optional): List of tasks to reset. 
                                    If None, resets all tasks.
        """
        tasks_to_reset = tasks if tasks else TASK_MARKERS.keys()
        for task in tasks_to_reset:
            marker = TASK_MARKERS.get(task)
            if marker and os.path.exists(marker):
                os.remove(marker)
                self.logger.info(f"Reset marker for task '{task}': {marker}")
            else:
                self.logger.debug(f"No marker found for task '{task}' — nothing to reset")

