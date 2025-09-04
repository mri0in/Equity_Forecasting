import io
import types
import yaml
import pytest
import importlib
from types import SimpleNamespace

import src.utils.model_utils as mu


class DummyModel:
    """Simple dummy model for testing instantiation."""

    def __init__(self, params=None):
        self.params = params or {}


def test_load_config_success(tmp_path):
    """load_config should correctly parse YAML file into dict."""
    cfg = {"key": "value", "num": 42}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    result = mu.load_config(str(cfg_path))
    assert result == cfg
    assert isinstance(result, dict)


def test_load_config_failure(tmp_path):
    """load_config should raise exception if file is invalid."""
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(":\t this is not valid yaml")

    with pytest.raises(Exception):
        mu.load_config(str(bad_path))


def test_instantiate_model_success(monkeypatch):
    """instantiate_model should dynamically import and return instance."""
    cfg = {
        "module": "fake.module",
        "class": "DummyModel",
        "params": {"lr": 0.01},
    }

    fake_module = types.SimpleNamespace(DummyModel=DummyModel)  # behaves like a module
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    model = mu.instantiate_model(cfg)
    assert isinstance(model, DummyModel)
    assert model.params == {"lr": 0.01}

def test_instantiate_model_missing_class(monkeypatch):
    """instantiate_model should raise if class is missing."""
    cfg = {"module": "fake.module", "class": "NoSuchModel"}

    monkeypatch.setattr(importlib, "import_module", lambda _: {})

    with pytest.raises(Exception):
        mu.instantiate_model(cfg)


def test_instantiate_model_invalid_module(monkeypatch):
    """instantiate_model should raise if import fails."""
    cfg = {"module": "bad.module", "class": "DummyModel"}

    def bad_import(_):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", bad_import)

    with pytest.raises(Exception):
        mu.instantiate_model(cfg)


def test_load_config_and_model(monkeypatch, tmp_path):
    """load_config_and_model should return (config, model)."""
    cfg = {
        "model": {"module": "fake.module", "class": "DummyModel", "params": {"x": 1}},
        "extra": 123,
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    fake_module = types.SimpleNamespace(DummyModel=DummyModel)  # behaves like a module
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)


    config, model = mu.load_config_and_model(str(cfg_path))
    assert config["extra"] == 123
    assert isinstance(model, DummyModel)
    assert model.params == {"x": 1}
