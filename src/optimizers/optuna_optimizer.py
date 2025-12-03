# src/training/optimizers/optuna_optimizer.py
import optuna
from typing import Dict, Any
from src.models.lstm_model import LSTMModel
from src.utils.logger import get_logger
from src.monitoring.monitor import log_stage_start, log_stage_end

logger = get_logger(__name__)

def objective(trial: optuna.Trial, config: Dict[str, Any], X_train, y_train) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial (optuna.Trial): Optuna trial object
        config (dict): Base configuration dictionary
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets

    Returns:
        float: Validation loss (e.g., RMSE)
    """
    log_stage_start(f"optuna_trial_{trial.number}", details={"trial_number": trial.number})

    # Example hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Update config with trial hyperparameters
    model_params = config.get("model_params", {}).copy()
    model_params.update({
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "dropout": dropout,
    })

    model = LSTMModel(model_params=model_params)

    # Train model
    model.train(X_train, y_train)

    # Predict on training set (or holdout if available)
    y_pred = model.predict(X_train)

    # Compute RMSE
    rmse = float(((y_train - y_pred) ** 2).mean() ** 0.5)
    logger.info(f"Trial {trial.number}: RMSE={rmse:.5f}, params={model_params}")

    log_stage_end(f"optuna_trial_{trial.number}", metrics={"rmse": rmse})
    return rmse


def run_optimization(config: Dict[str, Any], X_train, y_train, n_trials: int = 50) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization with stage monitoring.

    Args:
        config (dict): Base config dictionary
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        n_trials (int): Number of Optuna trials

    Returns:
        optuna.Study: Completed Optuna study object
    """
    log_stage_start("optuna_optimization", details={"n_trials": n_trials})

    study = optuna.create_study(direction="minimize")
    func = lambda trial: objective(trial, config, X_train, y_train)
    study.optimize(func, n_trials=n_trials)

    logger.info(f"Optuna optimization completed: Best trial {study.best_trial.number} with value {study.best_value}")
    logger.info(f"Best params: {study.best_trial.params}")

    log_stage_end("optuna_optimization", metrics={"best_value": study.best_value, "best_params": study.best_trial.params})
    return study