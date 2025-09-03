import pytest
import yaml
import tempfile
import os
from src.utils import config_loader

# -------------------------------------------------------------------------
# Tests for load_config
# -------------------------------------------------------------------------

def test_load_config_valid_yaml():
    """Test loading a valid YAML config file."""
    temp_config = {"model": {"module": "src.models.lstm_model", "class": "LSTMModel"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(temp_config, f)
        config_path = f.name

    loaded_config = config_loader.load_config(config_path)
    assert loaded_config["model"]["class"] == "LSTMModel"

    os.remove(config_path)


def test_load_config_invalid_yaml(tmp_path):
    """Test loading an invalid YAML file raises error."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: [unbalanced brackets")

    with pytest.raises(yaml.YAMLError):
        config_loader.load_config(str(config_file))


def test_load_config_missing_file():
    """Test loading a missing YAML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        config_loader.load_config("non_existent.yaml")

# -------------------------------------------------------------------------
# Tests for instantiate_model
# -------------------------------------------------------------------------

class DummyModel:
    """Simple dummy model for testing instantiation."""
    def __init__(self, param1: int, param2: str):
        self.param1 = param1
        self.param2 = param2


def test_instantiate_model_valid(monkeypatch):
    """Test instantiating a valid model dynamically."""
    monkeypatch.setattr("importlib.import_module", lambda _: __import__(__name__))

    model_config = {
        "module": __name__,
        "class_name": "DummyModel",
        "params": {"param1": 10, "param2": "test"}
    }

    model = config_loader.instantiate_model(model_config)
    assert isinstance(model, DummyModel)
    assert model.param1 == 10
    assert model.param2 == "test"


def test_instantiate_model_invalid_module():
    """Test instantiation fails if module does not exist."""
    model_config = {"module": "nonexistent.module", "class_name": "DummyModel"}
    with pytest.raises(ModuleNotFoundError):
        config_loader.instantiate_model(model_config)


def test_instantiate_model_invalid_class(monkeypatch):
    """Test instantiation fails if class does not exist in module."""
    monkeypatch.setattr("importlib.import_module", lambda _: __import__(__name__))

    model_config = {"module": __name__, "class_name": "NonExistentModel"}
    with pytest.raises(AttributeError):
        config_loader.instantiate_model(model_config)
