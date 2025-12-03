import numpy as np
import pytest
from unittest.mock import MagicMock

from src.pipeline.D_optimization_pipeline import load_training_data, run_hyperparameter_optimization


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

    # patch load_config
    monkeypatch.setattr("src.pipeline.run_optimizer.load_config", lambda _: mock_config)

    # patch np.load
    monkeypatch.setattr("numpy.load", lambda _: np.array([[1, 2], [3, 4]]))

    # patch get_optimizer
    mock_optimizer = MagicMock(return_value="fake_study")
    monkeypatch.setattr("src.pipeline.run_optimizer.get_optimizer", lambda *_: mock_optimizer)
    
    run_hyperparameter_optimization("fake_config.yaml", optimizer_name="optuna")

    # Check it was called exactly once
    mock_optimizer.assert_called_once()

    # Then manually check arguments
    args, kwargs = mock_optimizer.call_args
    assert args[0] == mock_config
    np.testing.assert_array_equal(args[1], np.array([[1, 2], [3, 4]]))  # X_train
    np.testing.assert_array_equal(args[2], np.array([[1, 2], [3, 4]]))  # y_train
    assert kwargs["n_trials"] == 10

def test_run_hyperparameter_optimization_no_training_key(monkeypatch):
    """Fallback to n_trials=50 if 'training' key is missing"""
    mock_config = {
        "data": {"X_train_path": "fake_X.npy", "y_train_path": "fake_y.npy"}
    }

    monkeypatch.setattr("src.pipeline.run_optimizer.load_config", lambda _: mock_config)
    monkeypatch.setattr("numpy.load", lambda _: np.array([1, 2, 3]))

    mock_optimizer = MagicMock(return_value="fake_study")
    monkeypatch.setattr("src.pipeline.run_optimizer.get_optimizer", lambda *_: mock_optimizer)

    run_hyperparameter_optimization("fake_config.yaml", optimizer_name="optuna")

    # Check it was called exactly once
    mock_optimizer.assert_called_once()

    
    args, kwargs = mock_optimizer.call_args
    assert args[0] == mock_config
    np.testing.assert_array_equal(args[1], np.array([1, 2, 3]))  # X_train
    np.testing.assert_array_equal(args[2], np.array([1, 2, 3]))  # y_train
    assert kwargs["n_trials"] == 50

def test_run_hyperparameter_optimization_with_other_optimizer(monkeypatch):
    """Custom optimizer and n_trials from config"""
    mock_config = {
        "data": {"X_train_path": "fake_X.npy", "y_train_path": "fake_y.npy"},
        "training": {"n_trials": 5},
    }

    monkeypatch.setattr("src.pipeline.run_optimizer.load_config", lambda _: mock_config)
    monkeypatch.setattr("numpy.load", lambda _: np.array([1, 2, 3]))

    mock_optimizer = MagicMock(return_value="fake_study")
    monkeypatch.setattr(
        "src.pipeline.run_optimizer.get_optimizer",
        lambda name: mock_optimizer if name == "raytune" else None,
    )

    run_hyperparameter_optimization("fake_config.yaml", optimizer_name="raytune")

    args, kwargs = mock_optimizer.call_args
    assert args[0] == mock_config
    np.testing.assert_array_equal(args[1], np.array([1, 2, 3]))  # X_train
    np.testing.assert_array_equal(args[2], np.array([1, 2, 3]))  # y_train
    assert kwargs["n_trials"] == 5

