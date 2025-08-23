import numpy as np
import pytest

from src.ensemble.simple_ensemble import simple_average, weighted_average


def test_simple_average():
    """Test that simple average computes correct mean across models."""
    preds = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.5, 2.5, 3.5]),
        np.array([2.0, 3.0, 4.0])
    ]

    result = simple_average(preds)

    expected = np.array([1.5, 2.5, 3.5])
    np.testing.assert_array_almost_equal(result, expected)


def test_weighted_average_correct():
    """Test that weighted average applies correct weighting."""
    preds = [
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0, 4.0])
    ]
    weights = [0.25, 0.75]

    result = weighted_average(preds, weights)

    expected = np.array([1.75, 2.75, 3.75])
    np.testing.assert_array_almost_equal(result, expected)


def test_weighted_average_invalid_weights():
    """Test that mismatched weights length raises ValueError."""
    preds = [np.array([1.0, 2.0]), np.array([2.0, 3.0])]
    weights = [0.3]  # Wrong length

    with pytest.raises(ValueError):
        weighted_average(preds, weights)


def test_ensemble_mismatched_shapes():
    """Test that mismatched prediction shapes raise ValueError."""
    preds = [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])]

    with pytest.raises(ValueError):
        simple_average(preds)
