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

        Fully defines all model parameters here, including static
        and sampled hyperparameters, so LSTMModel can be agnostic.
        """

        stage = f"optuna_trial_{trial.number}"
        self.monitor.log_stage_start(stage, {"trial_number": trial.number})

        # ------------------------
        # Static / fixed parameters
        # ------------------------
        input_size = X_train.shape[-1]
        batch_size = self.config.get("batch_size", 64)
        epochs = self.config.get("epochs", 10)
        num_layers = self.config.get("num_layers", 2)
        output_size = 1             # always predicting single target

        # ------------------------
        # Optuna-sampled parameters
        # ------------------------
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        # ------------------------
        # Consolidate model parameters
        # ------------------------
        model_params: Dict[str, Any] = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "output_size": output_size,
        }

        # ------------------------
        # Instantiate and train LSTM
        # ------------------------
        model = LSTMModel(model_params=model_params)
        model.train(X_train, y_train)

        # ------------------------
        # Evaluate
        # ------------------------
        y_pred = model.predict(X_train)
        rmse = float(np.sqrt(((y_train - y_pred) ** 2).mean()))

        logger.info(
            "[OPTUNA] Trial=%d RMSE=%.6f params=%s",
            trial.number,
            rmse,
            model_params,
        )

        self.monitor.log_stage_end(stage, {"rmse": rmse})
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
        Execute Optuna optimization and return best trial summary.
        """

        self.monitor.log_stage_start("optuna_optimization", {"n_trials": self.n_trials})

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: self._objective(t, X_train, y_train), n_trials=self.n_trials)

        result = {
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "best_trial": study.best_trial.number,
        }

        logger.info("[OPTUNA] Optimization completed | %s", result)
        self.monitor.log_stage_end("optuna_optimization", result)
        return result
