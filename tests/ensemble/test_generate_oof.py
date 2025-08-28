import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.ensemble.generate_oof import OOFGenerator


@pytest.fixture
def dummy_data():
    """Provide dummy X and y arrays for testing."""
    X = np.arange(20).reshape(10, 2)   # 10 samples, 2 features
    y = np.arange(10)                  # 10 targets
    return X, y


def test_load_data_success(tmp_path):
    """Test successful loading of .npy files."""
    x_path = tmp_path / "X.npy"
    y_path = tmp_path / "y.npy"
    X = np.arange(6).reshape(3, 2)
    y = np.array([1, 2, 3])

    np.save(x_path, X)
    np.save(y_path, y)

    generator = OOFGenerator(model_params={})
    X_loaded, y_loaded = generator.load_data(str(x_path), str(y_path))

    assert np.array_equal(X, X_loaded)
    assert np.array_equal(y, y_loaded)


def test_load_data_failure():
    """Test exception when file is missing."""
    generator = OOFGenerator(model_params={})
    with pytest.raises(Exception):
        generator.load_data("missing_X.npy", "missing_y.npy")


@patch("src.ensemble.generate_oof.LSTMModel")
def test_generate_with_mocked_model(mock_lstm, dummy_data):
    """Test OOF generation with mocked LSTMModel."""
    X, y = dummy_data
    mock_model_instance = MagicMock()
    mock_model_instance.predict.side_effect = lambda X_fold: np.array([99] * len(X_fold))
    mock_lstm.return_value = mock_model_instance

    generator = OOFGenerator(model_params={}, n_splits=2)
    preds, targets = generator.generate(X, y)

    # Ensure shapes match
    assert len(preds) == len(targets)
    assert set(targets).issubset(set(y))
    mock_model_instance.train.assert_called()


@patch("src.ensemble.generate_oof.np.save")
@patch("src.ensemble.generate_oof.os.makedirs")
def test_save_oof(mock_makedirs, mock_save, dummy_data, tmp_path):
    """Test saving OOF predictions and targets."""
    preds, targets = dummy_data[0][:, 0], dummy_data[1]
    out_dir = tmp_path / "oof"

    generator = OOFGenerator(model_params={})
    generator.save_oof(preds, targets, str(out_dir))

    mock_makedirs.assert_called_once_with(str(out_dir), exist_ok=True)
    # Check that np.save was called for preds and targets
    assert mock_save.call_count == 2
    assert any("oof_preds.npy" in call.args[0] for call in mock_save.call_args_list)
    assert any("oof_targets.npy" in call.args[0] for call in mock_save.call_args_list)
