# src/api/forecasting_api.py
"""
API wrapper used by the dashboard (ForecastPanel) to request a forecast
for a single equity.

Uses the new Orchestrator:
    orchestrator.run_forecast_pipeline(equity, horizon, ...)

If no prediction artifact exists, returns a simulated forecast to keep UI responsive.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.pipeline.orchestrator import PipelineOrchestrator
from src.monitoring.monitor import TrainingMonitor

monitor = TrainingMonitor()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------------------------------------------------------
# SIMULATION FALLBACK (UI-compatible)
# -------------------------------------------------------------------------
def _simulated_forecast(equity: str, horizon: int) -> Dict[str, Any]:
    """
    Return a simulated fallback forecast when no prediction artifact exists.
    """
    np.random.seed(abs(hash(equity)) % (2**32))

    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    hist_prices = np.linspace(150, 180, 30) + np.random.normal(0, 2.0, 30)

    last = float(hist_prices[-1])
    mean_step = np.random.normal(0, 0.5, horizon)
    forecast_mean = last + np.cumsum(mean_step)

    scale = np.linspace(0.5, 2.0, horizon)
    forecast_upper = forecast_mean + scale
    forecast_lower = forecast_mean - scale

    return {
        "equity": equity,
        "hist_dates": dates.astype(str).tolist(),
        "hist_prices": hist_prices.tolist(),
        "forecast_dates": (
            pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon)
        ).astype(str).tolist(),
        "forecast_mean": forecast_mean.tolist(),
        "forecast_upper": forecast_upper.tolist(),
        "forecast_lower": forecast_lower.tolist(),
        "forecast_prices": forecast_mean.tolist(),
        "metadata": {
            "source": "simulation",
            "model": None,
            "adapter": None,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }


# -------------------------------------------------------------------------
# ARTIFACT LOADER
# -------------------------------------------------------------------------
def _try_load_prediction_artifact(equity: str) -> Optional[Dict[str, Any]]:
    """
    Try to load prediction artifacts stored by the orchestrator.
    """

    candidate_paths = [
        f"datalake/predictions/{equity}_forecast.joblib",
        f"datalake/predictions/{equity}.joblib",
        f"datalake/cache/forecasting/{equity}_forecast.joblib",
        f"datalake/cache/forecasting/{equity}.joblib",
    ]

    for path in candidate_paths:
        if not os.path.exists(path):
            continue

        try:
            logger.info(f"Loading prediction artifact for {equity} from {path}")
            obj = joblib.load(path)

            # If dict → return as-is (canonical format)
            if isinstance(obj, dict):
                return obj

            # List/array fallback
            if isinstance(obj, (list, tuple, np.ndarray)):
                preds = list(obj)
                return {
                    "equity": equity,
                    "forecast_mean": preds,
                    "forecast_prices": preds,
                    "metadata": {"source": path},
                }

            # Unknown object → wrap it
            return {
                "equity": equity,
                "prediction": obj,
                "metadata": {"source": path},
            }

        except Exception as e:
            logger.error(f"Artifact load failed for {equity} from {path}: {e}")
            continue

    return None


# -------------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------------
def get_forecast_for_equity(
    equity: str,
    horizon: int,
    config_path: str = "config/config.yaml",
    force_rebuild_features: bool = False,
    allow_training: bool = False,
) -> Dict[str, Any]:
    """
    Main forecast entrypoint used by the dashboard/UI.

    Calls orchestrator.run_forecast_pipeline(), which handles:
        - feature preparation
        - prediction task
        - artifact generation & normalization
    """

    equity = equity.upper().strip()
    logger.info(f"Forecast request: equity={equity}, horizon={horizon}")

    monitor.log_stage_start("Forecast Request", {"equity": equity, "horizon": horizon})

    orchestrator = PipelineOrchestrator(config_path)

    # ------------------------------------------------------
    # (1) Optionally force rebuild of cached features
    # ------------------------------------------------------
    if force_rebuild_features:
        logger.info(f"Forcing feature rebuild for {equity}")
        try:
            orchestrator.prepare_features(equity)
        except Exception as e:
            logger.warning(f"Forced feature rebuild failed: {e}")

    # ------------------------------------------------------
    # (2) Run the orchestrator forecast pipeline
    # ------------------------------------------------------
    forecast = None
    try:
        forecast = orchestrator.run_forecast_pipeline(
            equity=equity,
            horizon=horizon,
            allow_training=allow_training,
        )
    except Exception as e:
        logger.error(f"orchestrator.run_forecast_pipeline failed: {e}")

    # If orchestrator already returned a usable forecast → we’re done
    if isinstance(forecast, dict):
        monitor.log_stage_end("Forecast Request", {"status": "orchestrator"})
        return forecast

    # ------------------------------------------------------
    # (3) Fallback: load artifact directly from disk
    # ------------------------------------------------------
    artifact = _try_load_prediction_artifact(equity)

    if artifact is not None:
        logger.info(f"Artifact found for {equity}, normalizing")

        result: Dict[str, Any] = {
            "equity": artifact.get("equity", equity),
            "metadata": artifact.get("metadata", {}),
        }

        # Numeric components
        if "forecast_mean" in artifact:
            result["forecast_mean"] = artifact["forecast_mean"]
            result["forecast_prices"] = artifact.get(
                "forecast_prices", artifact["forecast_mean"]
            )

        if "forecast_upper" in artifact:
            result["forecast_upper"] = artifact["forecast_upper"]
        if "forecast_lower" in artifact:
            result["forecast_lower"] = artifact["forecast_lower"]

        # Historical data
        if "hist_dates" in artifact:
            result["hist_dates"] = artifact["hist_dates"]
        if "hist_prices" in artifact:
            result["hist_prices"] = artifact["hist_prices"]

        # Forecast dates
        if "forecast_dates" in artifact:
            result["forecast_dates"] = artifact["forecast_dates"]
        else:
            today = pd.Timestamp.today()
            result["forecast_dates"] = (
                today + pd.to_timedelta(range(1, horizon + 1), unit="D")
            ).astype(str).tolist()

        # Metadata timestamp
        result["metadata"].setdefault(
            "generated_at",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

        monitor.log_stage_end("Forecast Request", {"status": "artifact"})
        return result

    # ------------------------------------------------------
    # (4) FINAL FALLBACK: simulated forecast
    # ------------------------------------------------------
    logger.warning(f"No artifact found for {equity}; returning simulated forecast")

    sim = _simulated_forecast(equity, horizon)

    monitor.log_stage_end("Forecast Request", {"status": "simulated"})
    return sim
