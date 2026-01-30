# src/api/forecasting_api.py
"""
Forecasting API used by the dashboard ForecastPanel.

Responsibilities:
- Collect inputs required by the adapter
- Call the adapter's public API (run_adapter_forecast)
- Normalize output for plotting
- Provide simulation fallback for UI resilience

"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any

from anyio import Path
import numpy as np
import pandas as pd

from src.adapter.adapter import run_adapter_forecast
from src.dashboard.history_manager import EquityHistory
from src.monitoring.monitor import TrainingMonitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

monitor = TrainingMonitor()


# ---------------------------------------------------------------------
# SIMULATION FALLBACK (UI SAFE)
# ---------------------------------------------------------------------
def _simulated_forecast(equity: str, horizon: int) -> Dict[str, Any]:
    """
    Generate a deterministic simulated forecast used only as UI fallback.
    """
    np.random.seed(abs(hash(equity)) % (2**32))

    hist_len = 30
    hist_dates = pd.date_range(end=pd.Timestamp.today(), periods=hist_len)
    hist_prices = np.linspace(150, 180, hist_len) + np.random.normal(0, 2.0, hist_len)

    last_price = float(hist_prices[-1])
    steps = np.random.normal(0, 0.5, horizon)
    forecast_prices = last_price + np.cumsum(steps)

    forecast_dates = pd.date_range(
        hist_dates[-1] + pd.Timedelta(days=1),
        periods=horizon,
    )

    return {
        "dates": (
            list(hist_dates.astype(str))
            + list(forecast_dates.astype(str))
        ),
        "hist_prices": hist_prices.tolist(),
        "forecast_prices": forecast_prices.tolist(),
        "metadata": {
            "source": "simulation",
            "equity": equity,
            "generated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        },
    }

def _load_Equity_Features(equity: str) -> pd.DataFrame:
    return EquityHistory().get_equity_data(equity)

def load_global_signal() -> np.ndarray:
    path = Path(__file__).resolve().parent / "global_signal" / "global_signal.npy" #temp Hardcoded path

    if not path.exists():
        raise FileNotFoundError("No active global signal promoted")
    
    signal = np.load(path)

    if signal.ndim != 1:
        raise ValueError("Global signal must be a 1D numpy array")

    return signal


# ---------------------------------------------------------------------
# PUBLIC API (USED BY DASHBOARD)
# ---------------------------------------------------------------------
def get_forecast_for_equity(
    equity: str,
    horizon: int,
    sentiment: float = 0.0,
) -> Dict[str, Any]:
    """
    Fetch a plot-ready forecast for a single equity.

    This function is called by the dashboard ForecastPanel.
    """

    equity = equity.upper().strip()
    logger.info("[API] Forecast request: equity=%s horizon=%s", equity, horizon)

    monitor.log_stage_start(
        "Forecast API",
        {"equity": equity, "horizon": horizon},
    )

    try:
        # ------------------------------------------------------------------
        # NOTE:
        # In current phase, equity_features, sentiment_snapshot, and
        # global_signal are assumed to be prepared upstream (cache / service).
        # These are injected here without pipeline execution.
        # ------------------------------------------------------------------

        equity_features = _load_Equity_Features(equity)      # placeholder / cache hook
        sentiment_snapshot = {"sentiment_score": sentiment}
        global_signal = load_global_signal()               # placeholder / cache hook

        adapter_result = run_adapter_forecast(
            active_equity=equity,
            equity_features=equity_features,
            sentiment_snapshot=sentiment_snapshot,
            global_signal=global_signal,
            horizon=horizon,
        )

        returns = np.array(adapter_result["return_forecast"])
        hist_prices = equity_features["close"].values
        last_price = hist_prices[-1]

        forecast_prices = last_price * np.cumprod(1 + returns)

        hist_len = len(hist_prices)
        forecast_len = len(forecast_prices)

        hist_dates = pd.date_range(
            end=pd.Timestamp.today(),
            periods=hist_len,
        )

        forecast_dates = pd.date_range(
            hist_dates[-1] + pd.Timedelta(days=1),
            periods=forecast_len,
        )

        result = {
            "dates": (
                list(hist_dates.astype(str))
                + list(forecast_dates.astype(str))
            ),
            "hist_prices": list(hist_prices),
            "forecast_prices": list(forecast_prices),
            "metadata": {
                "source": "adapter",
                "equity": equity,
                "generated_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            },
        }

        monitor.log_stage_end(
            "Forecast API",
            {"status": "adapter_success"},
        )

        return result

    except Exception as exc:
        logger.exception(
            "[API] Adapter failed for %s, using simulation fallback", equity
        )

        monitor.log_stage_end(
            "Forecast API",
            {"status": "simulation_fallback", "error": str(exc)},
        )

        return _simulated_forecast(equity, horizon)
