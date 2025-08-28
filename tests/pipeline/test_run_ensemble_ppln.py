# tests/pipeline/test_run_ensemble.py

import pytest
import numpy as np
import yaml
from unittest.mock import patch, MagicMock, mock_open

from src.pipeline.run_ensemble import run_ensemble


@pytest.fixture
def mock_config_mean(tmp_path):
    """Fixture for mean ensemble config with temporary paths."""
    y_true = np.array([1.0, 2.0, 3.0])
    pred1 = np.array([1.1, 2.1, 2.9])
    pred2 = np.array([0.9, 1.9, 3.2])

    # Save arrays to temporary .npy files
    y_true_path = tmp_path / "y_true.npy"
    pred1_path = tmp_path / "pred1.npy"
    pred2_path = tmp_path / "pred2.npy"

    np.save(y_true_path, y_true)
    np.save(pred1_path, pred1)
    np.save(pred2_path, pred2)

    return {
        "ensemble": {
            "method": "mean",
            "y_true_path": str(y_true_path),
            "pred_paths": [str(pred1_path), str(pred2_path)],
            "metrics": ["rmse", "mae"]
        }
    }


def test_load_ensemble_config(monkeypatch):
    """Test YAML config loading."""
    fake_yaml = {"ensemble": {"method": "mean"}}
    mock_open_func = mock_open(read_data=yaml.dump(fake_yaml))

    with patch("builtins.open", mock_open_func):
        cfg = run_ensemble.load_ensemble_config("fake_config.yaml")

    assert cfg["ensemble"]["method"] == "mean"


def test_run_ensemble_mean(mock_config_mean):
    """Test mean ensemble strategy end-to-end."""
    results = run_ensemble.run_ensemble(mock_config_mean)
    assert "rmse" in results
    assert "mae" in results
    assert results["rmse"] >= 0.0
    assert results["mae"] >= 0.0


def test_run_ensemble_weighted(monkeypatch, mock_config_mean):
    """Test weighted ensemble strategy with mocked np.load."""
    cfg = mock_config_mean
    cfg["ensemble"]["method"] = "weighted"
    cfg["ensemble"]["weights"] = [0.7, 0.3]

    results = run_ensemble.run_ensemble(cfg)
    assert "rmse" in results
    assert "mae" in results


def test_run_ensemble_stacked(monkeypatch):
    """Test stacked ensemble strategy with mocks for submodules."""

    cfg = {"ensemble": {"method": "stacked", "metrics": ["rmse", "mae"]}}

    with patch("src.pipeline.run_ensemble.generate_oof_predictions") as mock_gen, \
         patch("src.pipeline.run_ensemble.create_meta_features") as mock_meta, \
         patch("src.pipeline.run_ensemble.train_meta_model") as mock_train, \
         patch("src.pipeline.run_ensemble.evaluate_meta_model") as mock_eval:

        # Mock outputs
        mock_gen.return_value = ("oof_preds", "holdout_preds", np.array([1, 2, 3]))
        mock_meta.return_value = ("X_meta_train", "X_meta_test")
        mock_train.return_value = (np.array([1.0, 2.0, 3.0]), "fitted_model")
        mock_eval.return_value = {"rmse": 0.1, "mae": 0.05}

        results = run_ensemble.run_ensemble(cfg)

    assert results == {"rmse": 0.1, "mae": 0.05}


def test_run_ensemble_invalid_method():
    """Test unsupported ensemble method raises ValueError."""
    cfg = {"ensemble": {"method": "foobar"}}

    with pytest.raises(ValueError):
        run_ensemble.run_ensemble(cfg)
