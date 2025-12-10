# src/dag/testdag.py
"""
Mini Production-Like DAG Run
-----------------------------
Run a full DAG pipeline on a selected equity-set (local, CPU-friendly).
This script is intended for development / smoke testing: it loads an
equity-set YAML from src/dag/equity_sets/, injects the tickers into the
active-equity list, runs the DAG and then requests forecasts for each equity.

How to use:
- Swap `EQ_SET_FILENAME` to the set you want (eq_set_5.yaml, eq_set_50.yaml, ...)
- Run the script in your dev environment.

Notes:
- This script relies on the pipeline wrappers / PipelineStages implementation.
- All logging is to console (no file logging).
"""

from __future__ import annotations

import logging
import yaml
import torch
from pathlib import Path
from typing import List

from src.dag.dag_runner import DAGRunner
from src.dag.dag_graph import DAGGraph
from src.dag.state_manager import StateManager
from src.dag.dag_stages import PipelineStages
from src.config.active_equity import ACTIVE_EQUITIES
from src.predictor.predict import Predict

# -----------------------------
# Config: choose which equity-set YAML to run
# -----------------------------
EQ_SET_FILENAME = "eq_set_5.yaml"   # change to eq_set_50.yaml, eq_set_500.yaml, etc.
EQ_SET_DIR = Path(__file__).resolve().parents[0] / "equity_sets"
EQ_SET_PATH = EQ_SET_DIR / EQ_SET_FILENAME

# -----------------------------
# Logging setup (console only)
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("testdag")

# -----------------------------
# Device selection
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# -----------------------------
# Load equity set YAML
# -----------------------------
def load_equity_set(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Equity set file not found: {path}")
    with open(path, "r") as f:
        payload = yaml.safe_load(f)
    equities = payload.get("equities", [])
    if not isinstance(equities, list):
        raise TypeError("`equities` in equity-set YAML must be a list")
    return equities

try:
    test_equities = load_equity_set(EQ_SET_PATH)
    logger.info("Loaded equity set from %s: %s", EQ_SET_PATH, test_equities)
except Exception as e:
    logger.exception("Failed to load equity set: %s", e)
    raise

# -----------------------------
# Inject into ACTIVE_EQUITIES (so ingestion & pipelines can read it)
# -----------------------------
ACTIVE_EQUITIES.clear()
ACTIVE_EQUITIES.extend(test_equities)
logger.info("ACTIVE_EQUITIES set for this run: %s", list(ACTIVE_EQUITIES))

# -----------------------------
# DAG setup (ordered nodes)
# -----------------------------
dag = DAGGraph()
nodes = [
    "ingestion",
    "preprocessing",
    "feature_generation",
    "optimization",
    "ensembling",
    "training",
    "walkforward",
    "forecasting",
]
for node in nodes:
    dag.add_node(node)

# dependencies (topology)
dag.add_edge("ingestion", "preprocessing")
dag.add_edge("preprocessing", "feature_generation")
dag.add_edge("feature_generation", "optimization")
dag.add_edge("optimization", "ensembling")
dag.add_edge("ensembling", "training")
dag.add_edge("training", "walkforward")
dag.add_edge("walkforward", "forecasting")

# -----------------------------
# State manager and stages
# -----------------------------
state_manager = StateManager()
stages = PipelineStages(device=device)

# -----------------------------
# Run DAG
# -----------------------------
runner = DAGRunner(dag, state_manager, stages, max_retries=2, retry_delay_sec=5)

logger.info("Starting mini production-like DAG run...")
try:
    runner.run()
    logger.info("Mini DAG run completed successfully.")
except Exception as exc:
    logger.exception("Mini DAG run failed: %s", exc)
    raise

# -----------------------------
# Forecast equity-specific prices
# -----------------------------
predictor = Predict(device=device)

for equity in test_equities:
    try:
        forecast = predictor.get_forecast(equity)
        # Keep log concise: show last 5 rows (or brief message if empty)
        if hasattr(forecast, "tail"):
            logger.info("Forecast for %s (last 5 rows):\n%s", equity, forecast.tail(5))
        else:
            logger.info("Forecast for %s: %s", equity, str(forecast))
    except Exception as e:
        logger.exception("Failed to produce forecast for %s: %s", equity, e)
