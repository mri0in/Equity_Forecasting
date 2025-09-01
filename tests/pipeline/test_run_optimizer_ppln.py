import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.pipeline.run_optimizer import load_training_data,run_hyperparameter_optimization


def test_load_training_data_success(tmp_path):
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    x_file = tmp_path / "X.npy"
    y_file = tmp_path / "y.npy"
    np.save(x_file, X)
    np.save(y_file, y)

    loaded_X, loaded_y = load_training_data(str(x_file), str(y_file))
    assert np.array_equal(loaded_X, X)
    assert np.array_equal(loaded_y, y)


def test_load_training_data_failure(tmp_path):
    x_file = tmp_path / "X.npy"  # file does not exist
    y_file = tmp_path / "y.npy"  # file does not exist
    with pytest.raises(Exception):
        load_training_data(str(x_file), str(y_file))


def test_run_hyperparameter_optimization_default_optimizer(monkeypatch):
    # mock config
    mock_config = {
        "data": {"X_train_path": "fake_X.npy", "y_train_path": "fake_y.npy"},
        "training": {"n_trials": 10},
    }

    # patch load_yaml_config (not load_config)
    monkeypatch.setattr("src.pipeline.run_optimizer.load_config", lambda _: mock_config)

    # patch np.load
    monkeypatch.setattr("numpy.load", lambda _: np.array([[1, 2], [3, 4]]))

    # patch get_optimizer
    mock_optimizer = MagicMock(return_value="fake_study")
    monkeypatch.setattr("src.pipeline.run_optimizer.get_optimizer", lambda *_: mock_optimizer)
    
    # run function
    run_hyperparameter_optimization("fake_config.yaml", optimizer_name="optuna")

    # assert optimizer was called with expected args
    mock_optimizer.assert_called_once_with(
        mock_config,
        np.array([[1, 2], [3, 4]]),
        np.array([[1, 2], [3, 4]]),
        n_trials=10,
    )

def test_run_hyperparameter_optimization_no_training_key(monkeypatch):
    """Test when config has no 'training' key -> should fallback to default n_trials=50"""
    mock_config = {
        "data": {"X_train_path": "fake_X.npy", "y_train_path": "fake_y.npy"}
    }

    # patch load_config to return our mock config
    monkeypatch.setattr("src.pipeline.run_optimizer.load_config", lambda _: mock_config)

    # patch numpy.load to return dummy arrays
    monkeypatch.setattr("numpy.load", lambda _: np.array([1, 2, 3]))

    # patch optimizer factory
    mock_optimizer = MagicMock(return_value="fake_study")
    monkeypatch.setattr("src.pipeline.run_optimizer.get_optimizer", lambda _: mock_optimizer)

    run_hyperparameter_optimization("fake_config.yaml", optimizer_name="optuna")

    mock_optimizer.assert_called_once_with(
        mock_config, np.array([1, 2, 3]), np.array([1, 2, 3]), n_trials=50
    )


def test_run_hyperparameter_optimization_with_other_optimizer(monkeypatch):
    """Test with a different optimizer and custom n_trials in config"""
    mock_config = {
        "data": {"X_train_path": "fake_X.npy", "y_train_path": "fake_y.npy"},
        "training": {"n_trials": 5},
    }

    monkeypatch.setattr("src.pipeline.run_optimizer.load_config", lambda _: mock_config)
    monkeypatch.setattr("numpy.load", lambda _: np.array([1, 2, 3]))

    mock_optimizer = MagicMock(return_value="fake_study")
    monkeypatch.setattr("src.pipeline.run_optimizer.get_optimizer",lambda name: mock_optimizer if name == "raytune" else None)

    run_hyperparameter_optimization("fake_config.yaml", optimizer_name="raytune")

    mock_optimizer.assert_called_once_with(
        mock_config, np.array([1, 2, 3]), np.array([1, 2, 3]), n_trials=5
    )