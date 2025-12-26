# src/optimizers/optuna_optimizer.py
from __future__ import annotations

from typing import Dict, Any

import optuna
import numpy as np

from src.models.lstm_model import LSTMModel
from src.monitoring.monitor import TrainingMonitor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer.

    This optimizer is self-contained and intended to be invoked
    directly by Pipeline D (Optimization Pipeline).

    Responsibilities:
    - Define Optuna objective
    - Run study
    - Log trials and final results via TrainingMonitor

    Non-responsibilities:
    - Feature generation
    - Data splitting
    - Model persistence
    """

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        monitor: TrainingMonitor,
        n_trials: int,
    ) -> None:
        self.config = config
        self.monitor = monitor
        self.n_trials = n_trials

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> float:
        """
        Optuna objective function.
        """

        stage = f"optuna_trial_{trial.number}"
        self.monitor.log_stage_start(
            stage,
            {"trial_number": trial.number},
        )

        learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        )
        hidden_size = trial.suggest_int("hidden_size", 16, 128)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        model_params = self.config.get("model_params", {}).copy()
        model_params.update(
            {
                "learning_rate": learning_rate,
                "hidden_size": hidden_size,
                "dropout": dropout,
            }
        )

        model = LSTMModel(model_params=model_params)
        model.train(X_train, y_train)

        y_pred = model.predict(X_train)
        rmse = float(np.sqrt(((y_train - y_pred) ** 2).mean()))

        logger.info(
            "[OPTUNA] Trial=%d RMSE=%.6f params=%s",
            trial.number,
            rmse,
            model_params,
        )

        self.monitor.log_stage_end(
            stage,
            {"rmse": rmse},
        )

        return rmse

    # ------------------------------------------------------------------
    # Public API (Pipeline D entrypoint)
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Execute Optuna optimization.

        Returns
        -------
        Dict[str, Any]
            Best trial summary.
        """

        self.monitor.log_stage_start(
            "optuna_optimization",
            {"n_trials": self.n_trials},
        )

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda t: self._objective(t, X_train, y_train),
            n_trials=self.n_trials,
        )

        result = {
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "best_trial": study.best_trial.number,
        }

        logger.info("[OPTUNA] Optimization completed | %s", result)

        self.monitor.log_stage_end(
            "optuna_optimization",
            result,
        )

        return result
