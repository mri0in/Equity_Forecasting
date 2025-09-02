import numpy as np
import pytest
from unittest.mock import MagicMock

import src.training.walk_forward_validator as wfv


@pytest.fixture
def dummy_data():
    X = np.arange(20).reshape(-1, 1)   # 20 samples, 1 feature
    y = np.arange(20)                  # targets 0..19
    return X, y


def test_invalid_window_type_raises(dummy_data):
    model = MagicMock()
    with pytest.raises(ValueError):
        wfv.WalkForwardValidator(model=model, window_type="invalid")


def test_validate_expanding(monkeypatch, dummy_data):
    X, y = dummy_data

    # Fake model
    model = MagicMock()
    model.predict.return_value = np.ones(1)

    # Fake compute_metrics
    def fake_compute(y_true, y_pred, metrics):
        return {m: float(len(y_true)) for m in metrics}
    monkeypatch.setattr(wfv, "compute_metrics", fake_compute)

    validator = wfv.WalkForwardValidator(
        model=model,
        window_type="expanding",
        window_size=5,
        step_size=5,
        metrics=["rmse", "mae"]
    )
    result = validator.validate(X, y)

    # Each fold returns metrics with value equal to test size (here = 5)
    # Averaged -> still 5.0
    assert result["rmse"] == pytest.approx(5.0)
    assert result["mae"] == pytest.approx(5.0)
    model.train.assert_called()  # trained at least once


def test_validate_rolling(monkeypatch, dummy_data):
    X, y = dummy_data

    model = MagicMock()
    model.predict.return_value = np.zeros(2)

    monkeypatch.setattr(
        wfv, "compute_metrics",
        lambda y_true, y_pred, metrics: {m: 42.0 for m in metrics}
    )

    validator = wfv.WalkForwardValidator(
        model=model,
        window_type="rolling",
        window_size=10,
        step_size=2,
        metrics=["rmse"]
    )
    result = validator.validate(X, y)
    assert "rmse" in result
    assert result["rmse"] == 42.0


def test_validate_handles_model_error(monkeypatch, dummy_data):
    X, y = dummy_data

    # Model that raises in train
    model = MagicMock()
    model.train.side_effect = RuntimeError("boom")

    monkeypatch.setattr(
        wfv, "compute_metrics",
        lambda y_true, y_pred, metrics: {m: 1.0 for m in metrics}
    )

    validator = wfv.WalkForwardValidator(model=model, window_size=5)
    result = validator.validate(X, y)

    # Because training always fails -> no metrics
    assert result == {}


def test_average_metrics_empty():
    model = MagicMock()
    validator = wfv.WalkForwardValidator(model=model)
    avg = validator._average_metrics([])
    assert avg == {}
