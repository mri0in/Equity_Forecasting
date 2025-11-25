# src/api/forecasting_api.py
"""
API wrapper used by the dashboard (ForecastPanel) to request a forecast
for a single equity.

Now aligned with the NEW orchestration design:

    - Calls: orchestrator.run_forecast_pipeline(equity, horizon, ...)
    - Expects orchestrator to write canonical artifacts at:

        datalake/predictions/{equity}_forecast.joblib

If no artifact is found, returns a simulated forecast to keep UI responsive.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------------------------------------------------------
# SIMULATION FALLBACK
# -------------------------------------------------------------------------
def _simulated_forecast(equity: str, horizon: int) -> Dict[str, Any]:
    """
    Generate fallback simulated forecast (UI-compatible structure).
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
    Load forecast artifact for the given equity.

    """

    candidate_paths = [
        f"datalake/predictions/{equity}_forecast.joblib",
        f"datalake/predictions/{equity}.joblib",
        f"datalake/cache/forecasting/{equity}_forecast.joblib",
        f"datalake/cache/forecasting/{equity}.joblib",
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Loading prediction artifact for {equity} from {path}")
                obj = joblib.load(path)

                # If canonical dict, return directly
                if isinstance(obj, dict):
                    return obj

                # Minimal wrapper for list/numpy
                if isinstance(obj, (list, tuple, np.ndarray)):
                    preds = list(obj)
                    return {
                        "equity": equity,
                        "forecast_mean": preds,
                        "forecast_prices": preds,
                        "metadata": {"source": path},
                    }

                # Unknown type fallback
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
    Main forecast entrypoint for dashboard/UI.

    NOW uses the new orchestrator.run_forecast_pipeline().
    """

    equity = equity.upper().strip()
    logger.info(f"Forecast request: equity={equity}, horizon={horizon}")

    orchestrator = PipelineOrchestrator(config_path)

    # Step 1: optional forced feature rebuild
    if force_rebuild_features:
        try:
            logger.info(f"Forcing feature rebuild for {equity}")
            orchestrator.prepare_features(equity)
        except Exception as e:
            logger.warning(f"Forced feature rebuild failed: {e}")

    # Step 2: run forecast pipeline using NEW orchestration
    forecast = None
    try:
       forecast = orchestrator.run_forecast_pipeline(
            equity=equity,
            horizon=horizon,
            allow_training=allow_training,
        )
    except Exception as e:
        logger.error(f"orchestrator.run_forecast_pipeline failed: {e}")

    
    # If orchestrator returned a forecast dict → USE IT
    if isinstance(forecast, dict):
        logger.info("Using orchestrator-produced forecast (no reload needed).")
        return forecast    

    # Step 3: load artifact from disk (fallback)
    artifact = _try_load_prediction_artifact(equity)

    if artifact is not None:
        logger.info(f"Artifact found for {equity}, normalizing output.")

        result: Dict[str, Any] = {
            "equity": artifact.get("equity", equity),
            "metadata": artifact.get("metadata", {}),
        }

        # Forecast values
        if "forecast_mean" in artifact:
            result["forecast_mean"] = artifact["forecast_mean"]
            result["forecast_prices"] = artifact.get(
                "forecast_prices", artifact["forecast_mean"]
            )

        if "forecast_upper" in artifact:
            result["forecast_upper"] = artifact["forecast_upper"]
        if "forecast_lower" in artifact:
            result["forecast_lower"] = artifact["forecast_lower"]

        # Dates
        if "hist_dates" in artifact:
            result["hist_dates"] = artifact["hist_dates"]
        if "hist_prices" in artifact:
            result["hist_prices"] = artifact["hist_prices"]

        if "forecast_dates" in artifact:
            result["forecast_dates"] = artifact["forecast_dates"]
        else:
            today = pd.Timestamp.today()
            result["forecast_dates"] = (
                today + pd.to_timedelta(range(1, horizon + 1), unit="D")
            ).astype(str).tolist()

        # Ensure metadata
        result["metadata"].setdefault(
            "generated_at",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

        return result

    # Step 4: fallback simulation
    logger.warning(f"No artifact found for {equity}; returning simulated forecast")
    return _simulated_forecast(equity, horizon)

