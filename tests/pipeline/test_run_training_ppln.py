import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.E_model_trainer_pipeline import ModelTrainerPipeline, run_training_pipeline

def test_load_training_data_success(tmp_path, monkeypatch):
    # Prepare fake data files
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    x_file = tmp_path / "X.npy"
    y_file = tmp_path / "y.npy"
    np.save(x_file, X)
    np.save(y_file, y)

    # Patch config to point to these files
    fake_config = {"data": {"X_train_path": str(x_file), "y_train_path": str(y_file)}}
    fake_model = MagicMock()
    monkeypatch.setattr("src.pipeline.run_training.load_config_and_model", lambda _: (fake_config, fake_model))

    pipeline = ModelTrainerPipeline("fake_config.yaml")
    loaded_X, loaded_y = pipeline.load_training_data()

    np.testing.assert_array_equal(loaded_X, X)
    np.testing.assert_array_equal(loaded_y, y)


def test_load_training_data_failure(monkeypatch):
    # Patch config to invalid paths
    fake_config = {"data": {"X_train_path": "nonexistent_X.npy", "y_train_path": "nonexistent_y.npy"}}
    fake_model = MagicMock()
    monkeypatch.setattr("src.pipeline.run_training.load_config_and_model", lambda _: (fake_config, fake_model))

    pipeline = ModelTrainerPipeline("fake_config.yaml")

    with pytest.raises(Exception):
        pipeline.load_training_data()


def test_run_training_pipeline_success(monkeypatch):
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    fake_save_path = "fake_model.pkl"

    # Patch config/model
    fake_model = MagicMock()
    fake_model.train = MagicMock()
    fake_model.save_model = MagicMock()
    fake_config = {
        "data": {"X_train_path": "X.npy", "y_train_path": "y.npy"},
        "training": {"save_path": fake_save_path}
    }

    # Patch load_config_and_model
    monkeypatch.setattr("src.pipeline.run_training.load_config_and_model", lambda _: (fake_config, fake_model))
    # Patch numpy.load to return dummy data
    monkeypatch.setattr("numpy.load", lambda path: X if "X" in path else y)

    pipeline = ModelTrainerPipeline("fake_config.yaml")
    pipeline.run()

    # Ensure training and save were called
    fake_model.train.assert_called_once_with(X, y)
    fake_model.save_model.assert_called_once_with(fake_save_path)


def test_run_training_pipeline_failure(monkeypatch):
    # Patch config/model
    fake_model = MagicMock()
    fake_model.train.side_effect = RuntimeError("Training failed")
    fake_config = {
        "data": {"X_train_path": "X.npy", "y_train_path": "y.npy"},
        "training": {"save_path": "fake_model.pkl"}
    }

    monkeypatch.setattr("src.pipeline.run_training.load_config_and_model", lambda _: (fake_config, fake_model))
    monkeypatch.setattr("numpy.load", lambda path: np.array([1, 2, 3]))

    pipeline = ModelTrainerPipeline("fake_config.yaml")
    with pytest.raises(RuntimeError):
        pipeline.run()


def test_run_training_pipeline_function(monkeypatch):
    # Patch the pipeline class
    fake_pipeline = MagicMock()
    monkeypatch.setattr("src.pipeline.run_training.ModelTrainerPipeline", lambda config_path: fake_pipeline)

    run_training_pipeline("fake_config.yaml")
    fake_pipeline.run.assert_called_once()
