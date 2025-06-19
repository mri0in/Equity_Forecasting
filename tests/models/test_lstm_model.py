import os
import shutil
import json
import numpy as np
import torch
import pytest
import tempfile
from typing import Any, Dict, Optional


from src.models.lstm_model import LSTMModel

# Dummy model params for quick tests
MODEL_PARAMS = {
    "input_size": 3,
    "hidden_size": 8,
    "num_layers": 1,
    "output_size": 1,
    "dropout": 0.1,
    "batch_size": 4,
    "epochs": 2,
    "learning_rate": 0.01,
}

def train(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
    """
    Wrapper to satisfy abstract method 'train' from BaseModel.
    """
    return super().train(X_train, y_train, X_val, y_val)



def save_model(self, path: str) -> None:
    """
    Save model (BaseModel abstract requirement).
    """
    self.save(path)

@classmethod
def load_model(cls, path: str) -> "LSTMModel":
    """
    Load model (BaseModel abstract requirement).
    """
    return cls.load(path)

def predict(self, X_test: np.ndarray) -> np.ndarray:
    """
    Wrapper to satisfy abstract method 'predict' from BaseModel.
    """
    return super().predict(X_test)

def get_params(self) -> Dict[str, Any]:
    """
    Return model parameters (BaseModel abstract requirement).
    """
    return self.model_params


# Create dummy time series data
def create_dummy_data(samples=10, seq_len=5, input_size=3, output_size=1):
    X = np.random.randn(samples, seq_len, input_size).astype(np.float32)
    y = np.random.randn(samples, output_size).astype(np.float32)
    return X, y

@pytest.fixture
def model():
    return LSTMModel(model_params=MODEL_PARAMS)

# === Test Case: TC20250616_LstmModel_001 ===
# Description : Test that LSTMModel initializes with correct architecture and parameters.
# Component   : src/models/lstm_model.py
# Category    : Unit
# Author      : Mri
# Created On  : 2025-06-16 22:06
def test_model_initialization(model):
    # Act
    returned_params = model.get_params()

    # Assert
    assert isinstance(returned_params, dict)
    for key in MODEL_PARAMS:
        assert key in returned_params
        assert returned_params[key] == MODEL_PARAMS[key]

# === Test Case: TC20250616_LstmModel_002 ===
# Description : Train the model on dummy data and ensure predictions are generated correctly.
# Component   : src/models/lstm_model.py
# Category    : Unit
# Author      : Mri
# Created On  : 2025-06-16 22:10
def test_model_train_and_predict(model):
    # Step 1: Make a copy of original model params
    test_params = MODEL_PARAMS.copy()

    # Step 2: Inject temporary early stopping path into copied config
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_params["early_stopping"] = {
            "enabled": True,
            "patience": 2,
            "delta": 0.001,
            "checkpoint_path": os.path.join(tmp_dir, "best_model.pt")
        }

        # Step 3: Instantiate model
        model = LSTMModel(model_params=test_params)

        # Step 4: Train with dummy data
        X_train, y_train = create_dummy_data()
        X_val, y_val = create_dummy_data()
        model.train(X_train, y_train, X_val, y_val)

        # Step 5: Predict on test set
        X_test, _ = create_dummy_data(samples=3)
        preds = model.predict(X_test)

        # Step 6: Assertions
        assert preds.shape == (3, MODEL_PARAMS["output_size"])
        assert isinstance(preds, np.ndarray)
        
# === Test Case: TC20250616_LstmModel_004 ===
# Description : 
# Component   : src/models/lstm_model.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-16 22:13
def test_model_save_and_load(tmp_path):
    """
    Test saving and loading model with weights and params.
    """
    save_dir = tmp_path / "lstm_test_model"
    os.makedirs(save_dir, exist_ok=True)

    # Train a dummy model
    model = LSTMModel(MODEL_PARAMS)
    X_train, y_train = create_dummy_data()
    model.train(X_train, y_train)

    # Save model
    model.save_model(str(save_dir))
    assert os.path.exists(save_dir / "lstm_weights.pt")
    assert os.path.exists(save_dir / "model_params.json")

    # Load model
    loaded_model = LSTMModel.load_model(str(save_dir))
    assert isinstance(loaded_model, LSTMModel)
    assert loaded_model.model_params == MODEL_PARAMS

    # Ensure loaded model gives output
    X_test, _ = create_dummy_data(samples=2)
    preds = loaded_model.predict(X_test)
    assert preds.shape == (2, MODEL_PARAMS["output_size"])
