from __future__ import annotations

from typing import Dict, Any, Optional

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
    - Track best trained model
    - Log trials and results via TrainingMonitor

    Non-responsibilities:
    - Feature generation
    - Data splitting
    - Model persistence (handled by pipeline)
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

        # best-trial state
        self.best_model: Optional[LSTMModel] = None
        self.best_value: Optional[float] = None
        self.best_params: Optional[Dict[str, Any]] = None

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

        Trains an LSTM model and returns RMSE.
        """

        stage = f"optuna_trial_{trial.number}/{self.n_trials-1}"
        self.monitor.log_stage_start(stage, {"trial_number": trial.number})

        # ------------------------
        # Static parameters (YAML-controlled)
        # ------------------------
        input_size = X_train.shape[-1]
        output_size = 1

        batch_size = int(self.config.get("batch_size", 64))
        epochs = int(self.config.get("epochs", 10))
        num_layers = int(self.config.get("num_layers", 1))
        lookback = int(self.config.get("model", {}).get("lookback", 10))
        
        # ------------------------
        # Optuna-sampled parameters
        # ------------------------
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
        )
        hidden_size = trial.suggest_int(
            "hidden_size", 32, 256, step=32
        )
        dropout = trial.suggest_float(
            "dropout", 0.0, 0.3
        )

        # ------------------------
        # Consolidated model params
        # ------------------------
        model_params: Dict[str, Any] = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "lookback": lookback,
            "output_size": output_size,
        }

        # ------------------------
        # Train model
        # ------------------------
        model = LSTMModel(model_params=model_params)
        model.train(X_train, y_train)

        # ------------------------
        # Evaluate (train RMSE)
        # ------------------------
        y_pred = model.predict(X_train)
        rmse = float(np.sqrt(np.mean((y_train - y_pred) ** 2)))

        logger.info(
            "[OPTUNA] Trial=%d | RMSE=%.6f | params=%s",
            trial.number,
            rmse,
            model_params,
        )

        # ------------------------
        # Track best model
        # ------------------------
        if self.best_value is None or rmse < self.best_value:
            self.best_value = rmse
            self.best_params = model_params
            self.best_model = model

            logger.info(
                "[OPTUNA] New best model | trial=%d | RMSE=%.6f",
                trial.number,
                rmse,
            )

        self.monitor.log_stage_end(stage, {"rmse": rmse})
        return rmse

    # ------------------------------------------------------------------
    # Public API
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
        self.monitor.log_stage_end("optuna_optimization", result)
        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_best_model(self) -> LSTMModel:
        """
        Return the trained best model after optimization.
        """
        if self.best_model is None:
            raise RuntimeError("Best model not available. Did optimization run?")
        return self.best_model
