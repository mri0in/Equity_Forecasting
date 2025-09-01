import pytest
import numpy as np
import pandas as pd
import yaml
from unittest.mock import patch, MagicMock, mock_open

from src.pipeline.run_ensemble import run_ensemble,load_ensemble_config


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
        cfg = load_ensemble_config("fake_config.yaml")

    assert cfg["ensemble"]["method"] == "mean"


def test_run_ensemble_mean(mock_config_mean):
    """Test mean ensemble strategy end-to-end."""
    results = run_ensemble(mock_config_mean)
    assert "rmse" in results
    assert "mae" in results
    assert results["rmse"] >= 0.0
    assert results["mae"] >= 0.0


def test_run_ensemble_weighted(monkeypatch, mock_config_mean):
    """Test weighted ensemble strategy with mocked np.load."""
    cfg = mock_config_mean
    cfg["ensemble"]["method"] = "weighted"
    cfg["ensemble"]["weights"] = [0.7, 0.3]

    results = run_ensemble(cfg)
    assert "rmse" in results
    assert "mae" in results

def test_run_ensemble_simple():
    """Test simple ensemble flow using mocks for all components."""

    cfg = {
        "ensemble": {
            "method": "mean",
            "metrics": ["rmse", "mae"],
            "pred_paths": [
                "outputs/preds/model1.csv",
                "outputs/preds/model2.csv"
            ],
            "y_true_path": "outputs/true/labels.csv"
        },
    }

    with patch("src.pipeline.run_ensemble.OOFGenerator") as MockOOFGen, \
         patch("src.pipeline.run_ensemble.SimpleEnsembler") as MockEnsembler, \
         patch("src.pipeline.run_ensemble.compute_metrics") as mock_metrics, \
         patch("src.pipeline.run_ensemble._load_pred_arrays") as mock_loader, \
         patch("numpy.load") as mock_npload:

        # Prevent OOFGenerator from touching disk
        MockOOFGen.return_value.generate.return_value = (
            "oof_preds",
            "holdout_preds",
            np.array([1, 2, 3])
        )

        # Prevent _load_pred_arrays from touching disk
        mock_loader.return_value = [np.array([1, 2, 3]), np.array([2, 2, 2])]

        # Return fake labels when y_true is loaded
        mock_npload.return_value = np.array([1, 2, 3])

        # Mock ensembler
        MockEnsembler.return_value.average.return_value = np.array([2.0, 2.0, 2.0])

        # Mock metrics
        mock_metrics.return_value = {"rmse": 0.2, "mae": 0.1}

        results = run_ensemble(cfg)

        assert results == {"rmse": 0.2, "mae": 0.1}

def test_run_ensemble_stacked():
    """Test stacked ensemble flow using mocks for all components."""

    cfg = {
        "ensemble": {
            "method": "stacked",
            "metrics": ["rmse", "mae"],
            "oof": {
                "X_path": "fake_X.npy",
                "y_path": "fake_y.npy",
                "n_splits": 3,
                "model_params": {"hidden_size": 64, "num_layers": 2},
                "out_dir": "datalake/ensemble/oof"
            },
        }
    }

    with patch("src.pipeline.run_ensemble.OOFGenerator") as MockOOFGen, \
         patch("src.pipeline.run_ensemble.MetaFeaturesBuilder") as MockMeta, \
         patch("src.pipeline.run_ensemble.MetaFeatureTrainer") as MockTrainer, \
         patch("src.pipeline.run_ensemble.compute_metrics") as mock_metrics:

        # Mock OOF generator
        mock_oof = MockOOFGen.return_value
        mock_oof.load_data.return_value = ("X_fake", "y_fake")
        mock_oof.generate.return_value = ("oof_preds", "holdout_preds")

        # Mock meta feature builder
        mock_meta = MockMeta.return_value
        mock_meta.build.return_value = ("X_meta_train", "X_meta_test")

        # Mock meta trainer
        mock_trainer = MockTrainer.return_value
        mock_trainer.prepare_datasets.return_value = ("X_train", "y_train", "X_val", "y_val")
        mock_trainer.train.return_value = np.array([1.0, 2.0, 3.0])

        # Mock metrics
        mock_metrics.return_value = {"rmse": 0.1, "mae": 0.05}
        mock_trainer.evaluate.return_value = {"rmse": 0.1, "mae": 0.05}
        results = run_ensemble(cfg)

    assert results == {"rmse": 0.1, "mae": 0.05}

def test_run_ensemble_invalid_method():
    """Test unsupported ensemble method raises ValueError."""
    cfg = {"ensemble": {"method": "foobar"}}

    with pytest.raises(ValueError):
        run_ensemble(cfg)
