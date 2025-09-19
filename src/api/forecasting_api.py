"""
Forecasting API router.

This module exposes endpoints for generating equity price forecasts.
It connects the API layer with the predictor module, ensuring clean
separation of concerns between request handling and business logic.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Dict

# Business logic imports (predictor handles actual forecasting)
#from src.predictor.forecast_runner import ForecastRunner

# ------------------------------------------------------------
# Router & Logger Setup
# ------------------------------------------------------------
router = APIRouter()
logger = logging.getLogger("forecasting_api")
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@router.get("/predict", response_model=Dict[str, float])
async def predict_equity(
    ticker: str = Query(..., description="Equity ticker symbol (e.g., RELIANCE, AAPL)"),
    horizon: int = Query(5, description="Forecast horizon in days"),
    simulate: bool = Query(False, description="Use simulated forecast if True"),
) -> Dict[str, float]:
    """
    Generate forecasts for a given equity ticker over the specified horizon.
    Can optionally return a simulated forecast for demo purposes.

    Args:
        ticker (str): Equity ticker symbol.
        horizon (int): Number of days to forecast ahead.
        simulate (bool): If True, return a simulated forecast instead of real model prediction.

    Returns:
        Dict[str, float]: Dictionary with forecasted values keyed by date (YYYY-MM-DD).
    """
    try:
        logger.info(f"Forecast request received: ticker={ticker}, horizon={horizon}, simulate={simulate}")

        runner = None #ForecastRunner(ticker=ticker, horizon=horizon)

        if simulate:
            forecast = runner.simulate_forecast()
            logger.info(f"Simulated forecast returned for {ticker}")
        else:
            forecast = runner.run_forecast()
            logger.info(f"Real forecast generated successfully for {ticker}")

        # Ensure forecast is a dict keyed by date strings
        return forecast

    except Exception as e:
        logger.error(f"Forecasting failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Forecasting failed")
