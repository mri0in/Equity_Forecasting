# src/dag/config.py
"""
DAG Configuration
-----------------

Defines pipeline stages, dependencies, and optional parameters for DAG execution.

This file serves as the central configuration for the DAG runner, 
allowing easy modification of the pipeline structure or stage order.
"""

from typing import Dict, List

# ----------------------------------------------------------------------
# Define all DAG nodes / stages
# ----------------------------------------------------------------------
STAGES: List[str] = [
    "ingestion",
    "preprocessing",
    "feature_generation",
    "training",
    "optimization",
    "walkforward",
    "ensembling",
    "forecasting",
]

# ----------------------------------------------------------------------
# Define dependencies for each stage
# ----------------------------------------------------------------------
# Key = stage name, Value = list of stages that must be completed before this stage
DEPENDENCIES: Dict[str, List[str]] = {
    "ingestion": [],
    "preprocessing": ["ingestion"],
    "feature_generation": ["preprocessing"],
    "training": ["feature_generation"],
    "optimization": ["training"],          # Optional: can run after training
    "walkforward": ["training"],           # Depends on trained model
    "ensembling": ["walkforward"],         # Depends on walk-forward metrics
    "forecasting": ["ensembling"],         # Final stage
}

# ----------------------------------------------------------------------
# Optional parameters for each stage (if needed)
# ----------------------------------------------------------------------
STAGE_PARAMS: Dict[str, dict] = {
    "ingestion": {"source": "yfinance", "cache_path": "datalake/cache/data_processor"},
    "preprocessing": {"fill_missing": True, "scaling": "standard"},
    "feature_generation": {"technical": True, "sentiment": True},
    "training": {"epochs": 10, "batch_size": 64, "device": "cpu"},
    "optimization": {"trials": 50},
    "walkforward": {"window_size": 252},  # ~1 trading year
    "ensembling": {"methods": ["simple", "meta"]},
    "forecasting": {"horizon_days": 5},
}
