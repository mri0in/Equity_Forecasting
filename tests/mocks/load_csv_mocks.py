# tests/mocks.py

from unittest.mock import MagicMock
import pandas as pd
import numpy as np

def get_mock_model():
    """
    Returns a MagicMock model with mocked fit() and predict().
    """
    model = MagicMock()
    model.fit.return_value = None
    model.predict.return_value = [0.1, 0.2, 0.3]
    return model

def get_fake_dataframe():
    """
    Returns a synthetic pandas DataFrame representing stock data.
    """
    return pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "open": [100.0, 102.0, 104.0],
        "high": [110.0, 112.0, 114.0],
        "low": [95.0, 97.0, 99.0],
        "close": [105.0, 107.0, 109.0],
        "volume": [1000000, 1200000, 1400000]
    })

def get_mock_response(json_data: dict):
    """
    Returns a mock HTTP response with the given JSON payload.
    """
    response = MagicMock()
    response.json.return_value = json_data
    response.raise_for_status.return_value = None
    return response

def get_mock_study():
    """
    Returns a mock Optuna study for hyperparameter optimization.
    """
    study = MagicMock()
    study.best_params = {"lr": 0.01, "batch_size": 32}
    study.best_value = 0.05
    return study
