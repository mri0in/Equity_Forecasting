import os
import tempfile
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Prevent GUI rendering in tests

import pytest

from src.ensemble import evaluate_meta_model as emm


def test_load_meta_features(tmp_path):
    # Create a fake CSV
    data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "target": [7, 8, 9]
    })
    file_path = tmp_path / "meta_features.csv"
    data.to_csv(file_path, index=False)

    X, y = emm.load_meta_features(str(file_path))

    assert X.shape == (3, 2)  # 3 samples, 2 features
    assert y.shape == (3,)
    assert np.array_equal(y, np.array([7, 8, 9]))


def test_evaluate_predictions_logs(caplog):
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])

    with caplog.at_level("INFO"):
        emm.evaluate_predictions(y_true, y_pred)

    assert "MAE" in caplog.text
    assert "RMSE" in caplog.text
    assert "MAPE" in caplog.text


def test_plot_predictions_runs_without_error():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    # Should not raise any error
    emm.plot_predictions(y_true, y_pred)


def test_load_model(tmp_path):
    # Create dummy model and save with joblib
    dummy_model = {"name": "meta_model", "version": 1}
    model_path = tmp_path / "model.pkl"
    joblib.dump(dummy_model, model_path)

    loaded_model = emm.load_model(str(model_path))

    assert isinstance(loaded_model, dict)
    assert loaded_model["name"] == "meta_model"
    assert loaded_model["version"] == 1
