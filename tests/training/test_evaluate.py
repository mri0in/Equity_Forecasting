import numpy as np
import pytest

from src.training.evaluate import (
    rmse,
    mae,
    mape,
    r2,
    sharpe_ratio,
    directional_accuracy,
    compute_metrics
)
testheader

def test_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert np.isclose(rmse(y_true, y_pred), 0.577, atol=1e-2)

def test_mae():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert mae(y_true, y_pred) == 0.3333333333333333

def test_mape():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    result = mape(y_true, y_pred)
    expected = np.mean(np.abs((y_true - y_pred) / y_true))
    assert np.isclose(result, expected)

def test_r2():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    assert np.isclose(r2(y_true, y_pred), 0.9486, atol=1e-3)

def test_sharpe_ratio_nonzero_std():
    returns = np.array([0.01, 0.02, -0.01, 0.03])
    ratio = sharpe_ratio(returns)
    assert isinstance(ratio, float)
    assert ratio != 0.0

def test_sharpe_ratio_zero_std():
    returns = np.array([0.01, 0.01, 0.01])
    assert sharpe_ratio(returns) == 0.0

def test_directional_accuracy():
    y_true = np.array([1, 2, 3, 2, 3])
    y_pred = np.array([1, 2, 2.5, 2, 3])
    acc = directional_accuracy(y_true, y_pred)
    assert 0.0 <= acc <= 1.0
    assert isinstance(acc, float)

def test_compute_metrics_selected():
    y_true = np.array([100, 105, 110, 120])
    y_pred = np.array([102, 106, 111, 118])
    metrics = compute_metrics(y_true, y_pred, ["rmse", "mae", "mape", "r2", "sharpe", "directional_accuracy"])

    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ["rmse", "mae", "mape", "r2", "sharpe", "directional_accuracy"])
    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["sharpe"], float)
