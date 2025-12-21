# src/utils/config_loader.py

import importlib
from pydantic import BaseModel
from typing import List, Optional
from src.utils.config import load_config  # your existing YAML loader


# -------------------
# Pydantic Configs
# -------------------
class DataConfig(BaseModel):
    X_train_path: str
    y_train_path: str

class ModelParams(BaseModel):
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout: float
    learning_rate: float
    batch_size: int
    num_epochs: int
    device: str

class ModelConfig(BaseModel):
    module: str
    class_name: str  # renamed to class_name to avoid reserved word 'class'
    params: ModelParams

class TrainingConfig(BaseModel):
    save_path: str

class EarlyStoppingConfig(BaseModel):
    patience: int
    delta: float
    checkpoint_path: str

class LoggingConfig(BaseModel):
    level: str
    file: Optional[str] = None

class PipelineConfig(BaseModel):
    tasks: List[str]
    retries: int = 1
    strict: bool = True
class FullConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    early_stopping: EarlyStoppingConfig
    logging: LoggingConfig
    pipeline: PipelineConfig


# -------------------
# Functions
# -------------------
def load_typed_config(config_path: str) -> FullConfig:
    """
    Load and validate the full config as a typed Pydantic model.

    Args:
        config_path (str): Path to main YAML config file.

    Returns:
        FullConfig: Typed configuration object.
    """
    raw_config = load_config(config_path)
    # Rename model 'class' key to 'class_name' for Pydantic model compatibility
    if 'model' in raw_config and 'class' in raw_config['model']:
        raw_config['model']['class_name'] = raw_config['model'].pop('class')
    return FullConfig(**raw_config)

def instantiate_model(model_config: dict):
    """
    Dynamically import and instantiate a model from config.

    Args:
        model_config (dict): Dictionary with 'module', 'class_name', and 'params'.

    Returns:
        object: Instantiated model.
    """
    module_name = model_config["module"]
    class_name = model_config["class_name"]
    params = model_config.get("params", {})

    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class(**params)