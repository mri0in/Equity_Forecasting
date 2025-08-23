import numpy as np
import pytest

from src.ensemble.meta_features import generate_meta_features


def test_meta_features_shape(dummy_model):
    """Test that meta-features have the correct shape."""
    X = np.random.rand(30, 5)
    y = np.random.rand(30)

    models = [dummy_model, dummy_model]  # Two identical dummy models
    meta_features = generate_meta_features(models, X, y, n_splits=3)

    assert meta_features.shape == (30, 2), "Meta-features should have shape (n_samples, n_models)"


def test_meta_features_reproducibility(dummy_model):
    """Test that meta-features are reproducible with the same random state."""
    X = np.random.rand(30, 5)
    y = np.random.rand(30)

    models = [dummy_model, dummy_model]

    mf1 = generate_meta_features(models, X, y, n_splits=3, random_state=123)
    mf2 = generate_meta_features(models, X, y, n_splits=3, random_state=123)

    assert np.allclose(mf1, mf2), "Meta-features should be reproducible"


def test_meta_features_invalid_splits(dummy_model):
    """Test that invalid n_splits raises a ValueError."""
    X = np.random.rand(12, 4)
    y = np.random.rand(12)

    models = [dummy_model]

    with pytest.raises(ValueError):
        generate_meta_features(models, X, y, n_splits=20)
