# src/utils/config_loader.py

from pydantic import BaseModel
from typing import Optional
from utils.config import load_config  # your existing YAML loader

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
    file: Optional[str]

class FullConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    early_stopping: EarlyStoppingConfig
    logging: LoggingConfig

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
