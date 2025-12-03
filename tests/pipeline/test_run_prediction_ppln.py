import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.pipeline.H_prediction_pipeline import load_features, run_prediction_pipeline


def test_load_features_success(tmp_path):
    # Create a fake CSV file
    data = pd.DataFrame([[1, 2], [3, 4]])
    feature_file = tmp_path / "features.csv"
    data.to_csv(feature_file, index=False)

    loaded = load_features(str(feature_file))
    np.testing.assert_array_equal(loaded, data.values)


def test_load_features_file_not_found(tmp_path):
    fake_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_features(str(fake_path))


def test_run_prediction_pipeline_success(monkeypatch, tmp_path):
    # Fake config and paths
    fake_config = {"predictions": {"test_dir": str(tmp_path)}}
    fake_paths = {"features": {"test_features": str(tmp_path / "features.csv")}}
    data = pd.DataFrame([[1, 2], [3, 4]])
    data.to_csv(fake_paths["features"]["test_features"], index=False)

    # Mock load_config
    monkeypatch.setattr("src.pipeline.run_prediction.load_config", lambda path: fake_config if "config" in path else fake_paths)

    # Mock ModelPredictor
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = np.array([0.1, 0.2])
    monkeypatch.setattr("src.pipeline.run_prediction.ModelPredictor", lambda **kwargs: mock_predictor)

    # Run prediction
    run_prediction_pipeline("config.yaml", "paths.yaml", split="test")

    # Check calls
    mock_predictor.predict.assert_called_once()
    mock_predictor.save_predictions.assert_called_once()
    save_path = os.path.join(fake_config["predictions"]["test_dir"], "predictions.csv")
    np.testing.assert_array_equal(mock_predictor.predict.call_args[0][0], data.values)
    #mock_predictor.save_predictions.assert_called_with(mock_predictor.save_predictions.call_args[1], save_path)
    called_args, called_kwargs = mock_predictor.save_predictions.call_args
    assert called_args[1] == save_path



def test_run_prediction_pipeline_missing_paths(monkeypatch, tmp_path):
    # Config or paths missing required keys
    fake_config = {"predictions": {}}
    fake_paths = {"features": {}}

    monkeypatch.setattr("src.pipeline.run_prediction.load_config", lambda path: fake_config if "config" in path else fake_paths)

    with pytest.raises(TypeError):
        run_prediction_pipeline("config.yaml", "paths.yaml", split="test")


def test_run_prediction_pipeline_validation_split(monkeypatch, tmp_path):
    # Check validation split works same as test
    fake_config = {"predictions": {"validation_dir": str(tmp_path)}}
    fake_paths = {"features": {"validation_features": str(tmp_path / "features.csv")}}
    data = pd.DataFrame([[1, 2]])
    data.to_csv(fake_paths["features"]["validation_features"], index=False)

    monkeypatch.setattr("src.pipeline.run_prediction.load_config", lambda path: fake_config if "config" in path else fake_paths)
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = np.array([0.5])
    monkeypatch.setattr("src.pipeline.run_prediction.ModelPredictor", lambda **kwargs: mock_predictor)

    run_prediction_pipeline("config.yaml", "paths.yaml", split="validation")
    mock_predictor.predict.assert_called_once()
    mock_predictor.save_predictions.assert_called_once()
