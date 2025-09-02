# tests/training/optimizers/test_optuna_optimizer.py

import pytest
import numpy as np
import optuna
from unittest.mock import MagicMock, patch
from src.training.optimizers import optuna_optimizer


@pytest.fixture
def fake_config():
    """Fixture for a fake config dictionary with default model parameters."""
    return {"model_params": {"learning_rate": 0.001, "hidden_size": 32, "dropout": 0.2}}


@pytest.fixture
def fake_data():
    """Fixture for small synthetic training data (X_train, y_train)."""
    X_train = np.random.rand(10, 5).astype(np.float32)  # 10 samples, 5 features
    y_train = np.random.rand(10).astype(np.float32)     # 10 target values
    return X_train, y_train


def test_objective_updates_params_and_returns_rmse(fake_config, fake_data):
    """
    Test that objective function:
    1. Updates config params with Optuna trial suggestions.
    2. Trains model and returns a valid RMSE float.
    """
    X_train, y_train = fake_data

    # Patch LSTMModel to avoid heavy computation
    with patch("src.training.optimizers.optuna_optimizer.LSTMModel") as MockModel:
        mock_model = MagicMock()
        # Predict perfectly -> RMSE = 0.0
        mock_model.predict.return_value = y_train
        MockModel.return_value = mock_model

        # Fake trial that returns controlled values
        trial = optuna.trial.FixedTrial({
            "learning_rate": 0.001,
            "hidden_size": 64,
            "dropout": 0.3
        })

        rmse = optuna_optimizer.objective(trial, fake_config, X_train, y_train)

        # Assertions
        assert isinstance(rmse, float)
        assert rmse == pytest.approx(0.0)  # Perfect predictions
        assert fake_config["model_params"]["hidden_size"] == 64
        assert fake_config["model_params"]["dropout"] == 0.3

def test_run_optimization_returns_study(fake_config, fake_data):
    """
    Test run_optimization:
    - Creates an Optuna study.
    - Runs optimization with patched objective that still suggests params.
    - Returns a Study object with best_trial and recorded params.
    """
    X_train, y_train = fake_data

    # Fake objective that records at least one param
    def fake_objective(trial, *args, **kwargs):
        trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        return 0.5

    with patch("src.training.optimizers.optuna_optimizer.objective", fake_objective):
        study = optuna_optimizer.run_optimization(fake_config, X_train, y_train, n_trials=2)

        # Assertions
        assert isinstance(study, optuna.Study)
        assert study.best_value == pytest.approx(0.5)
        assert "learning_rate" in study.best_trial.params  # Now recorded
