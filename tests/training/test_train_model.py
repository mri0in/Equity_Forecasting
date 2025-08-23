import os
import tempfile
import yaml
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.training.train_model import ModelTrainer, run_training
from tests.training.dummy_model import DummyModel


@pytest.fixture
def sample_config(tmp_path):
    """Fixture: creates a temporary YAML config for testing."""
    config = {
        "data": {
            "X_train_path": str(tmp_path / "X_train.npy"),
            "y_train_path": str(tmp_path / "y_train.npy"),
            "X_val_path": str(tmp_path / "X_val.npy"),
            "y_val_path": str(tmp_path / "y_val.npy"),
        },
        "model": {
            "module": "tests.training.dummy_model",
            "class": "DummyModel",
            "params": {
                "early_stopping": {"enabled": True}
            }
        },
        "training": {"save_path": str(tmp_path / "model.pkl")}
    }

    # Save config to YAML
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Create dummy numpy data
    np.save(config["data"]["X_train_path"], np.random.randn(10, 5))
    np.save(config["data"]["y_train_path"], np.random.randn(10))
    np.save(config["data"]["X_val_path"], np.random.randn(4, 5))
    np.save(config["data"]["y_val_path"], np.random.randn(4))

    return config, config_path

# === Test Case: TC20250823_TrainModel_01 ===
# Description : Test that config loading works correctly. 
# Component   : src/training/train_model.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-08-23 22:48
def test_load_config_success(sample_config):
    _, config_path = sample_config
    trainer = ModelTrainer(str(config_path))
    assert "data" in trainer.config
    assert "model" in trainer.config

# === Test Case: TC20250823_TrainModel_02 ===
# Description : Test that config loading raises FileNotFoundError for missing file.
# Component   : src/training/train_model.py
# Category    : Unit
# Author      : Mri
# Created On  : 2025-08-23 22:48
def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        ModelTrainer("non_existent.yaml")

# === Test Case: TC20250823_TrainModel_03 ===
# Description : Test loading data with validation paths.
# Component   : src/training/train_model.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-08-23 22:51
def test_load_data_with_validation(sample_config):
    _, config_path = sample_config
    trainer = ModelTrainer(str(config_path))
    X, y, X_val, y_val = trainer.load_data()
    assert X.shape[0] == 10
    assert X_val.shape[0] == 4

# === Test Case: TC20250823_TrainModel_04 ===
# Description : Test loading data without validation paths raises KeyError.
# Component   : src/training/train_model.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-08-23 22:52
def test_load_data_missing_val_paths(tmp_path, sample_config):
    config, config_path = sample_config
    del config["data"]["X_val_path"]
    del config["data"]["y_val_path"]

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    trainer = ModelTrainer(str(config_path))
    with pytest.raises(KeyError):
        trainer.load_data()

# === Test Case: TC20250823_TrainModel_05 ===
# Description : Test model initialization with DummyModel.
# Component   : src/training/train_model.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-08-23 22:54
def test_initialize_model(sample_config):
    _, config_path = sample_config
    trainer = ModelTrainer(str(config_path))
    assert isinstance(trainer.model, DummyModel)
    assert trainer.model.params["early_stopping"]["enabled"] is True

# === Test Case: TC20250823_TrainModel_06 ===
# Description : Test the full training and saving pipeline with mocks.
# Component   : src/training/train_model.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-08-23 22:54
def test_train_and_save(monkeypatch, sample_config):
    _, config_path = sample_config
    trainer = ModelTrainer(str(config_path))

    # Monkeypatch DummyModel methods
    trainer.model.train = MagicMock()
    trainer.model.save = MagicMock()

    trainer.train_and_save()
    trainer.model.train.assert_called_once()
    trainer.model.save.assert_called_once_with(
        trainer.config["training"]["save_path"]
    )

# === Test Case: TC20250823_TrainModel_07 ===
# Description : Test run_training function that invokes the training pipeline.
# Component   : src/training/train_model.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-08-23 22:55
def test_run_training_invokes_pipeline(sample_config, monkeypatch):
    _, config_path = sample_config

    mock_trainer = MagicMock()
    monkeypatch.setattr("src.training.train_model.ModelTrainer", lambda x: mock_trainer)

    run_training(str(config_path))
    mock_trainer.train_and_save.assert_called_once()
