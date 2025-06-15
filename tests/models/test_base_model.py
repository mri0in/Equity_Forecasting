import pytest
from src.models.base_model import BaseModel

# Dummy model to test abstract base class behavior
class DummyModel(BaseModel):
    def train(self, X, y, X_val=None, y_val=None): pass
    def predict(self, X): return []
    def get_params(self): return {}
    def save(self, path): pass
    def load(self, path): pass

# === Test Case: TC20250615_baseModel_001 ===
# Description : Test that BaseModel cannot be instantiated directly due to it being an abstract method.
# Component   : src/models/base_model.py
# Category    : Unit
# Author      : Mri
# Created On  : 2025-06-15 22:09
def test_base_model_is_abstract():

    with pytest.raises(TypeError):
        BaseModel()

# === Test Case: TC20250616_baseModel_002 ===
# Description : Test that BaseModel can be instantiated with parameters.
# Component   : src/models/base_model.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-16 00:36
def test_base_model_init():
    model = DummyModel({"early_stopping": {"enabled": True, "patience": 3}})
    assert model.early_stopping_enabled is True
    assert model.early_stopping_params["patience"] == 3


# === Test Case: TC20250615_baseModel_003 ===
# Description : Test only the default set early stopping configuration is triggered when no params are passed.
# Component   : src/models/base_model.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-06-15 22:11
def test_default_early_stopping_values():
    
    model = DummyModel()

    assert model.early_stopping_enabled is False
    assert model.early_stopping_params["patience"] == 5
    assert model.early_stopping_params["delta"] == 1e-4
    assert model.early_stopping_params["checkpoint_path"] == "checkpoints/best_model.pt"

# === Test Case: TC20250616_baseModel_004 ===
# Description : Test that custom ES(Early Stopping) config is parsed correctly and info level of logger message is generated.
# Component   : src/models/base_model.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-06-16 00:11
def test_custom_early_stopping_config(caplog):
    
    custom_params = {
        "early_stopping": {
            "enabled": True,
            "patience": 10,
            "delta": 0.002,
            "checkpoint_path": "custom/checkpoint.pt"
        }
    }

    with caplog.at_level("INFO"):
        model = DummyModel(model_params=custom_params)

    assert model.early_stopping_enabled is True
    assert model.early_stopping_params["patience"] == 10
    assert model.early_stopping_params["delta"] == 0.002
    assert model.early_stopping_params["checkpoint_path"] == "custom/checkpoint.pt"

    assert any("Early stopping enabled" in record for record in caplog.text.splitlines())
