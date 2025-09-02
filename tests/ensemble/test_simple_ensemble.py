import numpy as np
import pytest

from src.ensemble.simple_ensembler import SimpleEnsembler

# ---- Fixtures ----
@pytest.fixture
def sample_predictions():
    # Two models, each predicting 3 values
    return [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])]

@pytest.fixture
def y_true():
    return np.array([1.5, 2.5, 3.5])


# ---- Tests ----
def test_mean_ensemble(sample_predictions):
    ensembler = SimpleEnsembler(sample_predictions)
    result = ensembler.ensemble_predictions(method="mean")
    expected = np.mean(np.stack(sample_predictions, axis=0), axis=0)
    np.testing.assert_array_almost_equal(result, expected)


def test_median_ensemble(sample_predictions):
    ensembler = SimpleEnsembler(sample_predictions)
    result = ensembler.ensemble_predictions(method="median")
    expected = np.median(np.stack(sample_predictions, axis=0), axis=0)
    np.testing.assert_array_almost_equal(result, expected)


def test_weighted_ensemble(sample_predictions):
    weights = [0.7, 0.3]
    ensembler = SimpleEnsembler(sample_predictions)
    result = ensembler.ensemble_predictions(method="weighted", weights=weights)

    stacked = np.stack(sample_predictions, axis=0)
    expected = np.sum(stacked * np.array(weights).reshape(-1, 1), axis=0) / np.sum(weights)
    np.testing.assert_array_almost_equal(result, expected)


def test_weighted_invalid_weights(sample_predictions):
    # Mismatched length should raise ValueError
    ensembler = SimpleEnsembler(sample_predictions)  
    with pytest.raises(ValueError):
        ensembler.ensemble_predictions(method="weighted", weights=[0.5])# only one weight


def test_invalid_method(sample_predictions):
    ensembler = SimpleEnsembler(sample_predictions)
    with pytest.raises(ValueError):
        ensembler.ensemble_predictions(method="unknown")


def test_evaluate_monkeypatch(y_true, sample_predictions):
    # monkeypatch compute_metrics to control its output
    def fake_compute_metrics(y_t, y_p, metrics):
        return {"rmse": 0.123, "mae": 0.456, "metrics_used": metrics}

    ensembler = SimpleEnsembler(sample_predictions)

    # direct monkeypatch: replace the imported compute_metrics
    from src.ensemble import simple_ensembler
    simple_ensembler.compute_metrics = fake_compute_metrics

    y_pred = np.array([1.4, 2.6, 3.4])
    result = ensembler.evaluate(y_true, y_pred, metrics=["rmse", "mae"],
                                 compute_metrics=fake_compute_metrics)

    assert "rmse" in result
    assert "mae" in result
    assert result["metrics_used"] == ["rmse", "mae"]
