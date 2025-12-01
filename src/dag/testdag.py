# src/dag/mini_prod_run.py
"""
Mini Production-Like DAG Run
-----------------------------
Runs full DAG pipeline on 5 equities locally.
- Trains and caches global LSTM model
- Generates and caches features
- Runs optimizer, walk-forward, ensembling
- Produces equity-specific forecasts
"""

import logging
import torch
from src.dag.dag_runner import DAGRunner
from src.dag.dag_graph import DAGGraph
from src.dag.state_manager import StateManager
from src.dag.dag_stages import PipelineStages
from src.config.active_equity import ACTIVE_EQUITIES
from src.predictor.predict import Predict

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mini_prod_run")

# -----------------------------
# Device selection
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# -----------------------------
# Test equities
# -----------------------------
test_equities = ["AAPL", "MSFT", "GOOGL", "INFY", "TCS"]
ACTIVE_EQUITIES.clear()
ACTIVE_EQUITIES.extend(test_equities)
logger.info(f"Running pipeline for equities: {test_equities}")

# -----------------------------
# DAG setup
# -----------------------------
dag = DAGGraph()
nodes = [
    "ingestion",
    "preprocessing",
    "feature_generation",
    "training",
    "optimization",
    "walkforward",
    "ensembling",
    "forecasting",
]
for node in nodes:
    dag.add_node(node)

dag.add_edge("ingestion", "preprocessing")
dag.add_edge("preprocessing", "feature_generation")
dag.add_edge("feature_generation", "training")
dag.add_edge("training", "optimization")
dag.add_edge("training", "walkforward")
dag.add_edge("training", "ensembling")
dag.add_edge("ensembling", "forecasting")

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
runner.run()
logger.info("Mini DAG run completed.")

# -----------------------------
# Forecast equity-specific prices
# -----------------------------
predictor = Predict(device=device)
for equity in test_equities:
    forecast = predictor.get_forecast(equity)
    logger.info(f"Forecast for {equity} (last 5 entries):\n{forecast.tail(5)}")
