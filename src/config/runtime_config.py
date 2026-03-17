"""
Runtime configuration module.

This module centralizes environment-driven configuration for the
equity forecasting system. All filesystem paths and runtime settings
should be derived from environment variables defined here.
"""

import os
from pathlib import Path


class RuntimeConfig:
    """
    Central configuration object for runtime environment.
    """

    ENV: str = os.getenv("APP_ENV", "local")

    PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", ".")).resolve()

    DATA_ROOT: Path = PROJECT_ROOT / "datalake" / "data"
    CACHE_ROOT: Path = PROJECT_ROOT / "datalake" / "cache"
    RUNS_ROOT: Path = PROJECT_ROOT / "datalake" / "runs"

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


config = RuntimeConfig()