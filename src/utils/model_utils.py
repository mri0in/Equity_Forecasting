
"""
Utility functions for loading models and configuration for training or optimization.
"""

import yaml
import importlib
from typing import Tuple, Type
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file from disk.

    Args:
        config_path (str): Path to the YAML config file

    Returns:
        dict: Parsed configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info("Loaded config from %s", config_path)
            return config
    except Exception as e:
        logger.exception("Error loading config: %s", str(e))
        raise

def instantiate_model(model_config: dict):
    """
    Dynamically load and instantiate a model from its config.

    Args:
        model_config (dict): Dict with 'module', 'class', and 'params' keys

    Returns:
        object: Instantiated model class
    """
    try:
        module_path = model_config["module"]
        class_name = model_config["class"]
        params = model_config.get("params", {})

        module = importlib.import_module(module_path)
        model_class: Type = getattr(module, class_name)
        model = model_class(params)

        logger.info(f"Instantiated model {class_name} from {module_path}")
        return model
    except Exception as e:
        logger.exception("Failed to load model: %s", str(e))
        raise

def load_config_and_model(config_path: str) -> Tuple[dict, object]:
    """
    Load config and dynamically instantiate the model defined in it.

    Args:
        config_path (str): Path to YAML config

    Returns:
        Tuple[dict, object]: Parsed config and model instance
    """
    config = load_config(config_path)
    model = instantiate_model(config["model"])
    return config, model
