import numpy as np
import pytest

from src.ensemble.train_meta_features import train_meta_features
from tests.ensemble.conftest import DummyModel


def test_train_meta_features_output():
    """Test that training returns a fitted model and predictions of correct shape."""
    X_meta = np.random.rand(30, 4)
    y = np.random.rand(30)

    model, preds = train_meta_features(DummyModel(), X_meta, y)

    assert preds.shape == (30,), "Predictions should match number of samples"
    assert hasattr(model, "fit"), "Returned model should be a fitted estimator"


def test_train_meta_features_reproducibility():
    """Test that training with the same random state yields reproducible predictions."""
    X_meta = np.random.rand(25, 3)
    y = np.random.rand(25)

    _, preds1 = train_meta_features(DummyModel(), X_meta, y, random_state=123)
    _, preds2 = train_meta_features(DummyModel(), X_meta, y, random_state=123)

    assert np.allclose(preds1, preds2), "Predictions should be reproducible"


def test_train_meta_features_invalid_input():
    """Test that mismatched input shapes raise a ValueError."""
    X_meta = np.random.rand(20, 5)
    y = np.random.rand(15)  # wrong length

    with pytest.raises(ValueError):
        train_meta_features(DummyModel(), X_meta, y)
