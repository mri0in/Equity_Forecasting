import yaml
from typing import Any, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(config_path)
    if not path.is_file():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded config from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file: {e}")
        raise
