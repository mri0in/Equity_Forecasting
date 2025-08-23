# tests/ensemble/test_generate_oof.py

import numpy as np
import pytest

from src.ensemble.generate_oof import generate_oof


def test_generate_oof_shapes(dummy_model):
    """Test that OOF predictions have the correct shape."""
    X = np.random.rand(20, 5)
    y = np.random.rand(20)

    oof_preds = generate_oof(dummy_model, X, y, n_splits=5)

    assert oof_preds.shape == (20,), "OOF predictions should match number of samples"


def test_generate_oof_reproducibility(dummy_model):
    """Test that OOF predictions are reproducible with the same random state."""
    X = np.random.rand(20, 5)
    y = np.random.rand(20)

    preds1 = generate_oof(dummy_model, X, y, n_splits=4, random_state=42)
    preds2 = generate_oof(dummy_model, X, y, n_splits=4, random_state=42)

    assert np.allclose(preds1, preds2), "OOF predictions should be reproducible"


def test_generate_oof_invalid_splits(dummy_model):
    """Test that invalid n_splits raises a ValueError."""
    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    with pytest.raises(ValueError):
        generate_oof(dummy_model, X, y, n_splits=15)
