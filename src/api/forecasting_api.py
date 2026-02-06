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

from pathlib import Path
import numpy as np
import pandas as pd

from src.adapter.adapter import run_adapter_forecast
from src.dashboard.history_manager import EquityHistory
from src.monitoring.monitor import TrainingMonitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



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
    EquityHistory_Manager = EquityHistory()
    df = EquityHistory_Manager.get_equity_data(equity)
    return df

def load_global_signal() -> np.ndarray:
    
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "global_signal" / "global_signal.npy"

    if not path.exists():
        print("Global signal file not found at:", path)
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
    sentiment_simulated: bool = False,
    overall_sentiment: float = 0.0,
) -> Dict[str, Any]:
    """
    Fetch a plot-ready forecast for a single equity.

    This function is called by the dashboard ForecastPanel.
    
    Args:
        equity: Ticker symbol
        horizon: Forecast horizon in days
        sentiment: Sentiment score (used if sentiment_simulated is False)
        sentiment_simulated: Whether sentiment data is simulated
        overall_sentiment: Real overall sentiment score (used if not simulated)
    """

    equity = equity.upper().strip()
    logger.info("[API] Forecast request: equity=%s horizon=%s", equity, horizon)

    if not sentiment_simulated:
        logger.info("[API] Using real sentiment score: %s", overall_sentiment)
        sentiment = overall_sentiment


    try:
        # ------------------------------------------------------------------
        # NOTE:
        # In current phase, equity_features, sentiment_snapshot, and
        # global_signal are assumed to be prepared upstream (cache / service).
        # These are injected here without pipeline execution.
        # ------------------------------------------------------------------

        equity_features = _load_Equity_Features(equity)      
        sentiment_snapshot = {"sentiment_score": sentiment}
        global_signal = load_global_signal()                

        adapter_result = run_adapter_forecast(
            active_equity=equity,
            equity_features=equity_features,
            sentiment_snapshot=sentiment_snapshot,
            global_signal=global_signal,
            horizon=horizon,
        )

        returns = np.array(adapter_result["return_forecast"])
        
        # Log raw returns to understand scale
        logger.info(f"[API] Raw adapter returns: min={returns.min():.6f}, max={returns.max():.6f}, mean={returns.mean():.6f}")
        
        # Normalize returns to realistic percentage range (-5% to +5%)
        # Use sigmoid-like scaling to keep returns bounded
        returns_normalized = np.tanh(returns / 100) * 0.05  # Scales to ±5%
        
        logger.info(f"[API] Normalized returns: min={returns_normalized.min():.6f}, max={returns_normalized.max():.6f}, mean={returns_normalized.mean():.6f}")
        
        hist_prices = equity_features["close"].values
        last_price = hist_prices[-1]

        forecast_prices = last_price * np.cumprod(1 + returns_normalized)
        
        logger.info(f"[API] Forecast prices: min={forecast_prices.min():.2f}, max={forecast_prices.max():.2f}, last={forecast_prices[-1]:.2f}")

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


        return result

    except Exception as exc:
        logger.exception(
            "[API] Adapter failed for %s, using simulation fallback", equity
        )

        return _simulated_forecast(equity, horizon)
