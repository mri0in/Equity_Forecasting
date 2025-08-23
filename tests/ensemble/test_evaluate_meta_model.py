import numpy as np
import pytest

from src.ensemble.evaluate_meta_model import evaluate_meta_model


def test_evaluate_meta_model_metrics():
    """Test that evaluation returns expected metrics with simple data."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    results = evaluate_meta_model(y_true, y_pred)

    assert "mse" in results, "Evaluation results should include MSE"
    assert "rmse" in results, "Evaluation results should include RMSE"
    assert "r2" in results, "Evaluation results should include R2"
    assert results["mse"] >= 0, "MSE should be non-negative"
    assert -1 <= results["r2"] <= 1, "R2 should be between -1 and 1"


def test_evaluate_meta_model_reproducibility():
    """Test that evaluation on same inputs yields identical results."""
    y_true = np.random.rand(20)
    y_pred = np.random.rand(20)

    results1 = evaluate_meta_model(y_true, y_pred)
    results2 = evaluate_meta_model(y_true, y_pred)

    assert results1 == results2, "Evaluation results should be reproducible"


def test_evaluate_meta_model_invalid_input():
    """Test that mismatched lengths raise a ValueError."""
    y_true = np.random.rand(15)
    y_pred = np.random.rand(10)

    with pytest.raises(ValueError):
        evaluate_meta_model(y_true, y_pred)
