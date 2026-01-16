"""
Adapter Module — Equity Forecasting System
==========================================

Purpose
-------
This module implements the *Adapter* layer that bridges the global
market signal produced by the offline pipelines (A–H) with
user-specific, short-horizon equity forecasts.

The adapter is an inference-only component and is invoked exclusively
by the forecasting API (and indirectly by the dashboard). It conditions
local equity dynamics on:

1. A precomputed global market signal (ensemble output)
2. Latest sentiment snapshot for the active equity
3. Recent, preprocessed equity-specific features

Scope & Responsibilities
------------------------
- Fuse global + local + sentiment signals
- Produce short-term forecasts for a single equity
- Enforce strict input validation and alignment rules
- Return dashboard-ready outputs (no side effects)

Non-Responsibilities (STRICT)
-----------------------------
- No training or fine-tuning
- No persistence or datalake writes
- No pipeline or DAG interaction
- No feature engineering or preprocessing
- No model retraining or backtesting

Design Notes
------------
- This module is intentionally implemented as a SINGLE FILE
- The adapter logic is deterministic, fast, and interpretable
- Scaling and learning are handled upstream by pipelines A–H
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import logging

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================================
# Internal Helper Functions
# ======================================================================

def _validate_inputs(
    active_equity: str,
    equity_features: pd.DataFrame,
    sentiment_snapshot: Dict[str, Any],
    global_signal: np.ndarray,
    horizon: int,
) -> None:
    """
    Validate adapter inputs strictly and fail fast.
    """
    if not active_equity or not isinstance(active_equity, str):
        raise ValueError("active_equity must be a non-empty string")

    if not isinstance(equity_features, pd.DataFrame):
        raise TypeError("equity_features must be a pandas DataFrame")

    if equity_features.empty:
        raise ValueError("equity_features must not be empty")

    if not isinstance(sentiment_snapshot, dict):
        raise TypeError("sentiment_snapshot must be a dict")

    if not isinstance(global_signal, np.ndarray):
        raise TypeError("global_signal must be a numpy array")

    if global_signal.ndim != 1:
        raise ValueError("global_signal must be a 1D array")

    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")


def _align_series(
    equity_features: pd.DataFrame,
    global_signal: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Right-align equity features and global signal.

    Alignment Rule (FROZEN):
    -----------------------
    Use the most recent overlapping window between the two.
    No padding, no interpolation.
    """
    n_equity = len(equity_features)
    n_global = len(global_signal)

    n = min(n_equity, n_global)

    if n <= 0:
        raise ValueError("No overlapping data between equity features and global signal")

    return equity_features.iloc[-n:], global_signal[-n:]


def _extract_sentiment_score(sentiment_snapshot: Dict[str, Any]) -> float:
    """
    Extract a scalar sentiment score from the sentiment snapshot.

    This function is intentionally conservative: missing fields
    default to neutral sentiment.
    """
    score = sentiment_snapshot.get("sentiment_score", 0.0)

    try:
        return float(score)
    except Exception:
        logger.warning("Invalid sentiment_score value, defaulting to 0.0")
        return 0.0


# ======================================================================
# Adapter Model (Internal)
# ======================================================================

class AdapterModel:
    """
    Internal adapter model that fuses global, local, and sentiment signals.

    This is NOT a trainable model. All parameters are fixed heuristics
    designed for stability and interpretability.
    """

    def __init__(
        self,
        equity_weight: float = 0.40,
        global_weight: float = 0.45,
        sentiment_weight: float = 0.15,
    ):
        total = equity_weight + global_weight + sentiment_weight
        if not np.isclose(total, 1.0):
            raise ValueError("Adapter weights must sum to 1.0")

        self.equity_weight = equity_weight
        self.global_weight = global_weight
        self.sentiment_weight = sentiment_weight

    def predict(
        self,
        equity_features: pd.DataFrame,
        global_signal: np.ndarray,
        sentiment_score: float,
        horizon: int,
    ) -> np.ndarray:
        """
        Generate short-term forecasts.

        Strategy (v1):
        --------------
        - Equity signal: last observed return proxy
        - Global signal: recent mean of global latent state
        - Sentiment: additive bias
        """
        # Equity signal proxy: mean of last few rows
        equity_signal = equity_features.mean(axis=1).iloc[-1]

        # Global signal proxy: rolling mean of recent signal
        global_signal_value = float(np.mean(global_signal[-5:]))

        # Combine signals
        base_forecast = (
            self.equity_weight * equity_signal
            + self.global_weight * global_signal_value
            + self.sentiment_weight * sentiment_score
        )

        # Produce horizon-length forecast (flat, short-term assumption)
        forecast = np.full(shape=(horizon,), fill_value=base_forecast)

        return forecast


# ======================================================================
# Public API (ONLY ENTRYPOINT)
# ======================================================================

def run_adapter_forecast(
    active_equity: str,
    equity_features: pd.DataFrame,
    sentiment_snapshot: Dict[str, Any],
    global_signal: np.ndarray,
    horizon: int,
) -> Dict[str, Any]:
    """
    Run the adapter to produce short-term forecasts for a single equity.

    Parameters
    ----------
    active_equity : str
        Ticker symbol selected by the user.
    equity_features : pd.DataFrame
        Preprocessed and feature-engineered equity data (recent window).
    sentiment_snapshot : dict
        Latest sentiment metrics for the equity.
    global_signal : np.ndarray
        Global market signal produced by pipeline H.
    horizon : int
        Number of future steps to predict.

    Returns
    -------
    dict
        Dashboard-ready forecast payload.
    """
    logger.info("[ADAPTER] Starting forecast for equity=%s", active_equity)

    _validate_inputs(
        active_equity,
        equity_features,
        sentiment_snapshot,
        global_signal,
        horizon,
    )

    equity_features_aligned, global_signal_aligned = _align_series(
        equity_features,
        global_signal,
    )

    sentiment_score = _extract_sentiment_score(sentiment_snapshot)

    model = AdapterModel()

    predictions = model.predict(
        equity_features=equity_features_aligned,
        global_signal=global_signal_aligned,
        sentiment_score=sentiment_score,
        horizon=horizon,
    )

    confidence = float(
        max(0.0, min(1.0, 1.0 - np.std(predictions)))
    )

    result = {
        "equity": active_equity,
        "horizon": horizon,
        "predictions": predictions.tolist(),
        "confidence": confidence,
        "components": {
            "equity_signal_weight": model.equity_weight,
            "global_signal_weight": model.global_weight,
            "sentiment_weight": model.sentiment_weight,
        },
        "metadata": {
            "adapter_version": "v1",
        },
    }

    logger.info(
        "[ADAPTER] Forecast completed | equity=%s horizon=%d",
        active_equity,
        horizon,
    )

    return result
