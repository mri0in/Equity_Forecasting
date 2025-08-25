import pytest
import numpy as np
from ensemble.simple_ensembler import SimpleEnsembler

def test_init_defaults():
    ensembler = SimpleEnsembler()
    assert ensembler.method == "mean"
    assert ensembler.weights is None
    assert ensembler.metrics == ["rmse", "mae"]

def test_init_custom():
    ensembler = SimpleEnsembler(method="median", metrics=["mae"])
    assert ensembler.method == "median"
    assert ensembler.metrics == ["mae"]


def test_mean_ensemble():
    preds = [np.array([1, 2, 3]), np.array([2, 3, 4])]
    ensembler = SimpleEnsembler(method="mean")
    result = ensembler.ensemble_predictions(preds)
    np.testing.assert_array_equal(result, np.array([1.5, 2.5, 3.5]))

def test_median_ensemble():
    preds = [np.array([1, 10, 3]), np.array([2, 20, 4]), np.array([3, 30, 5])]
    ensembler = SimpleEnsembler(method="median")
    result = ensembler.ensemble_predictions(preds)
    np.testing.assert_array_equal(result, np.array([2, 20, 4]))

def test_weighted_ensemble_valid():
    preds = [np.array([1, 2]), np.array([3, 4])]
    ensembler = SimpleEnsembler(method="weighted", weights=[0.7, 0.3])
    result = ensembler.ensemble_predictions(preds)
    expected = (0.7 * np.array([1, 2]) + 0.3 * np.array([3, 4])) / (0.7 + 0.3)
    np.testing.assert_allclose(result, expected)

def test_weighted_ensemble_invalid_weights():
    preds = [np.array([1, 2]), np.array([3, 4])]
    ensembler = SimpleEnsembler(method="weighted", weights=[0.5])  # wrong length
    with pytest.raises(ValueError, match="Invalid or missing weights"):
        ensembler.ensemble_predictions(preds)

def test_invalid_method():
    preds = [np.array([1, 2]), np.array([3, 4])]
    ensembler = SimpleEnsembler(method="unknown")
    with pytest.raises(ValueError, match="Unsupported ensemble method"):
        ensembler.ensemble_predictions(preds)


def test_evaluate_returns_dict(monkeypatch):
    # Monkeypatch evaluate_metrics to control output
    def fake_metrics(y_true, y_pred, metrics):
        return {"rmse": 0.5, "mae": 0.2}

    monkeypatch.setattr("src.ensemble.simple_ensembler", fake_metrics)

    ensembler = SimpleEnsembler()
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    result = ensembler.evaluate(y_true, y_pred)

    assert isinstance(result, dict)
    assert "rmse" in result
    assert "mae" in result
    assert result["rmse"] == 0.5
    assert result["mae"] == 0.2
