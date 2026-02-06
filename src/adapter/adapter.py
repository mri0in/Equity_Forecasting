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

"""

from __future__ import annotations

from typing import Dict, Any
import logging

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def _validate_inputs(
    active_equity: str,
    equity_features: pd.DataFrame,
    sentiment_snapshot: Dict[str, Any],
    global_signal: np.ndarray,
    horizon: int,
) -> None:
    if not active_equity:
        raise ValueError("active_equity must be provided")

    if equity_features.empty:
        raise ValueError("equity_features must not be empty")

    #if not isinstance(global_signal, np.ndarray):
        raise TypeError("global_signal must be numpy array")

    if horizon <= 0:
        raise ValueError("horizon must be > 0")


def _extract_sentiment(sentiment_snapshot: Dict[str, Any]) -> float:
    try:
        return float(sentiment_snapshot.get("sentiment_score", 0.0))
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# Adapter Model (Deterministic)
# ----------------------------------------------------------------------

class AdapterModel:
    """
    Deterministic signal fusion model (no training).
    """

    def __init__(
        self,
        equity_weight: float = 0.40,
        global_weight: float = 0.45,
        sentiment_weight: float = 0.15,
    ):
        if not np.isclose(
            equity_weight + global_weight + sentiment_weight, 1.0
        ):
            raise ValueError("Adapter weights must sum to 1.0")

        self.equity_weight = equity_weight
        self.global_weight = global_weight
        self.sentiment_weight = sentiment_weight

    def predict_returns(
        self,
        equity_features: pd.DataFrame,
        global_signal: np.ndarray,
        sentiment_score: float,
        horizon: int,
    ) -> np.ndarray:
        """
        Produce horizon-length RETURN forecast.
        
        When sentiment is unavailable (score == 0.0), dynamically redistribute 
        the sentiment weight to equity and global signals proportionally.
        """

        numeric_features = equity_features.select_dtypes(include="number")

        if numeric_features.empty:
            raise ValueError("No numeric equity features available for forecasting")

        equity_signal = equity_features.mean(axis=1).iloc[-1]
        global_signal_value = float(np.mean(global_signal[-5:]))

        # Log signal inputs for debugging
        logger.info(f"[ADAPTER] Signals - equity: {equity_signal:.6f}, global: {global_signal_value:.6f}, sentiment: {sentiment_score:.6f}")

        # Dynamic reweighting when sentiment is unavailable
        if sentiment_score == 0.0:
            # Redistribute sentiment weight proportionally to equity and global
            total_available_weight = self.equity_weight + self.global_weight
            adjusted_equity_w = self.equity_weight / total_available_weight
            adjusted_global_w = self.global_weight / total_available_weight
            
            logger.info(f"[ADAPTER] Using reweighted signals (no sentiment): equity_w={adjusted_equity_w:.3f}, global_w={adjusted_global_w:.3f}")
            
            blended_return = (
                adjusted_equity_w * equity_signal
                + adjusted_global_w * global_signal_value
            )
        else:
            # Use original weighted blend when sentiment is available
            logger.info(f"[ADAPTER] Using full weighted blend: equity_w={self.equity_weight:.3f}, global_w={self.global_weight:.3f}, sentiment_w={self.sentiment_weight:.3f}")
            
            blended_return = (
                self.equity_weight * equity_signal
                + self.global_weight * global_signal_value
                + self.sentiment_weight * sentiment_score
            )

        logger.info(f"[ADAPTER] Blended return before broadcast: {blended_return:.6f}")
        return np.full(horizon, blended_return)


# ----------------------------------------------------------------------
# Public API (ONLY ENTRYPOINT)
# ----------------------------------------------------------------------

def run_adapter_forecast(
    active_equity: str,
    equity_features: pd.DataFrame,
    sentiment_snapshot: Dict[str, Any],
    global_signal: np.ndarray,
    horizon: int,
) -> Dict[str, Any]:
    """
    Produce RETURN forecasts for a single equity.

    NOTE:
    -----
    This function DOES NOT produce prices.
    """

    logger.info("[ADAPTER] Running adapter for %s", active_equity)

    _validate_inputs(
        active_equity,
        equity_features,
        sentiment_snapshot,
        global_signal,
        horizon,
    )

    sentiment_score = _extract_sentiment(sentiment_snapshot)

    model = AdapterModel()

    return_forecast = model.predict_returns(
        equity_features=equity_features,
        global_signal=global_signal,
        sentiment_score=sentiment_score,
        horizon=horizon,
    )

    return {
        "equity": active_equity,
        "return_forecast": return_forecast.tolist(),
        "confidence": float(max(0.0, 1.0 - np.std(return_forecast))),
        "metadata": {
            "adapter_version": "v1",
        },
    }
