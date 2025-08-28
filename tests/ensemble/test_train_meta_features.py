import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.ensemble.train_meta_features import MetaFeatureTrainer


@pytest.fixture
def dummy_csv(tmp_path):
    """Create a dummy CSV file with meta-features + target."""
    df = pd.DataFrame({
        "feat1": range(10),
        "feat2": range(10, 20),
        "target": range(20, 30)
    })
    file_path = tmp_path / "meta.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_init_with_invalid_path(tmp_path):
    """Ensure FileNotFoundError is raised for invalid path."""
    fake_path = tmp_path / "no_file.csv"
    with pytest.raises(FileNotFoundError):
        MetaFeatureTrainer(str(fake_path))


def test_init_with_invalid_test_size(dummy_csv):
    """Ensure ValueError is raised for invalid test_size."""
    with pytest.raises(ValueError):
        MetaFeatureTrainer(str(dummy_csv), test_size=2)


def test_prepare_datasets_success(dummy_csv):
    """Ensure prepare_datasets loads and splits correctly."""
    trainer = MetaFeatureTrainer(str(dummy_csv))
    X_train, y_train, X_val, y_val = trainer.prepare_datasets()

    assert not X_train.empty
    assert not X_val.empty
    assert all(col in ["feat1", "feat2"] for col in X_train.columns)
    assert y_train.name == "target"


def test_prepare_datasets_missing_target(tmp_path):
    """Raise ValueError if target column is missing."""
    df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    trainer = MetaFeatureTrainer(str(path))
    with pytest.raises(ValueError):
        trainer.prepare_datasets()


@patch("src.ensemble.train_meta_features.lgb.train")
@patch("src.ensemble.train_meta_features.lgb.Dataset")
def test_train_model_and_evaluate(mock_dataset, mock_train, dummy_csv):
    """Test training and evaluation with mocked LightGBM."""
    trainer = MetaFeatureTrainer(str(dummy_csv))
    X_train, y_train, X_val, y_val = trainer.prepare_datasets()

    # Mock Booster
    mock_booster = MagicMock()
    mock_booster.best_iteration = 1
    mock_booster.predict.return_value = np.array(y_val)  # perfect prediction
    mock_booster.feature_importance.return_value = np.array([10, 5])

    mock_train.return_value = mock_booster
    trainer.train_model(X_train, y_train, X_val, y_val)

    assert trainer.model is mock_booster
    rmse = trainer.evaluate(X_val, y_val)
    assert rmse == 0  # perfect prediction


def test_evaluate_without_training(dummy_csv):
    """Ensure evaluate raises if no model trained."""
    trainer = MetaFeatureTrainer(str(dummy_csv))
    X_train, y_train, X_val, y_val = trainer.prepare_datasets()

    with pytest.raises(ValueError):
        trainer.evaluate(X_val, y_val)


@patch("src.ensemble.train_meta_features.joblib.dump")
def test_save_model(mock_dump, dummy_csv):
    """Ensure save_model calls joblib.dump with trained model."""
    trainer = MetaFeatureTrainer(str(dummy_csv))
    trainer.model = MagicMock()  # fake trained model

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pkl")
        trainer.save_model(path)
        mock_dump.assert_called_once_with(trainer.model, path)


def test_save_model_without_training(dummy_csv):
    """Raise error if save_model called before training."""
    trainer = MetaFeatureTrainer(str(dummy_csv))
    with pytest.raises(ValueError):
        trainer.save_model("fake.pkl")
