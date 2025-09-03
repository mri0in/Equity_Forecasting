import types
import numpy as np
import pandas as pd
import pytest

from src.predictor.predict import ModelPredictor


# -----------------------
# Dummy model for testing
# -----------------------
class DummyModel:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def load_model(cls, checkpoint_path):
        return cls()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones((X.shape[0], 1))  # always returns 1


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def fake_configs(tmp_path, monkeypatch):
    """Create temporary YAML configs and patch dynamic import only for dummy_module."""
    config_path = tmp_path / "config.yaml"
    paths_path = tmp_path / "paths.yaml"
    checkpoint = tmp_path / "dummy.ckpt"
    checkpoint.write_text("checkpoint")

    config_content = {
        "model": {"module": "dummy_module", "class": "DummyModel"},
        "early_stopping": {"checkpoint_path": str(checkpoint)},
    }
    paths_content = {
        "predictions": {
            "test_dir": str(tmp_path / "test_preds"),
            "validation_dir": str(tmp_path / "val_preds"),
        }
    }

    import yaml, importlib, importlib.metadata

    config_path.write_text(yaml.safe_dump(config_content))
    paths_path.write_text(yaml.safe_dump(paths_content))

    # Create a proper fake module
    fake_module = types.ModuleType("dummy_module")
    fake_module.DummyModel = DummyModel
    fake_module.__version__ = "0.0.1"

    # Keep the real import for everything else
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "dummy_module":
            return fake_module
        return real_import(name, *args, **kwargs)

    # Patch only for dummy_module
    monkeypatch.setattr(importlib, "import_module", fake_import)

    return str(config_path), str(paths_path), str(checkpoint)

# -----------------------
# Tests
# -----------------------
def test_init_with_checkpoint(fake_configs):
    config_path, paths_path, ckpt = fake_configs
    predictor = ModelPredictor(config_path, paths_path, model_checkpoint=ckpt)
    assert isinstance(predictor.model, DummyModel)


def test_init_without_checkpoint_uses_config(fake_configs):
    config_path, paths_path, ckpt = fake_configs
    predictor = ModelPredictor(config_path, paths_path)  # uses config checkpoint
    assert isinstance(predictor.model, DummyModel)


def test_init_fails_if_checkpoint_missing(fake_configs, tmp_path):
    config_path, paths_path, _ = fake_configs
    bad_ckpt = tmp_path / "does_not_exist.ckpt"
    with pytest.raises(FileNotFoundError):
        ModelPredictor(config_path, paths_path, model_checkpoint=str(bad_ckpt))


def test_predict(fake_configs):
    config_path, paths_path, ckpt = fake_configs
    predictor = ModelPredictor(config_path, paths_path, model_checkpoint=ckpt)
    X = np.zeros((5, 3))
    preds = predictor.predict(X)
    assert preds.shape == (5, 1)
    assert np.all(preds == 1)


def test_save_predictions_1d(tmp_path, fake_configs):
    config_path, paths_path, ckpt = fake_configs
    predictor = ModelPredictor(config_path, paths_path, model_checkpoint=ckpt)
    preds = np.array([1, 2, 3])
    save_path = tmp_path / "preds.csv"
    predictor.save_predictions(preds, str(save_path))
    df = pd.read_csv(save_path)
    assert list(df.columns) == ["prediction"]
    assert df.shape[0] == 3


def test_save_predictions_2d(tmp_path, fake_configs):
    config_path, paths_path, ckpt = fake_configs
    predictor = ModelPredictor(config_path, paths_path, model_checkpoint=ckpt)
    preds = np.array([[1, 2], [3, 4]])
    save_path = tmp_path / "preds.csv"
    predictor.save_predictions(preds, str(save_path))
    df = pd.read_csv(save_path)
    assert list(df.columns) == ["pred_0", "pred_1"]
    assert df.shape == (2, 2)


def test_save_predictions_from_type_test(tmp_path, fake_configs):
    config_path, paths_path, ckpt = fake_configs
    predictor = ModelPredictor(config_path, paths_path, model_checkpoint=ckpt)

    # Patch paths to point inside tmp_path
    predictor.paths["predictions"]["test_dir"] = str(tmp_path)

    preds = np.array([1, 2, 3])
    predictor.save_predictions_from_type(preds, "test")
    df = pd.read_csv(tmp_path / "predictions.csv")
    assert "prediction" in df.columns


def test_save_predictions_from_type_invalid(fake_configs):
    config_path, paths_path, ckpt = fake_configs
    predictor = ModelPredictor(config_path, paths_path, model_checkpoint=ckpt)
    preds = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        predictor.save_predictions_from_type(preds, "invalid_split")
