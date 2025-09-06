"""
Sentiment API router.

This module exposes endpoints for retrieving sentiment analysis results
based on news, social media, and press/web feeds. It connects the API
layer with the market sentiment module.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any

# Business logic import (aggregates feeds + sentiment model)
from src.features.market_sentiment.processing.aggregator import SentimentAggregator
from src.config.active_equity import get_active_equity

# ------------------------------------------------------------
# Router & Logger Setup
# ------------------------------------------------------------
router = APIRouter()
logger = logging.getLogger("sentiment_api")


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@router.get("/analyze", response_model=Dict[str, Any])
async def analyze_sentiment(
    ticker: str = Query(..., description="Equity ticker symbol (e.g., RELIANCE, AAPL)"),
    source: str = Query("all", description="Feed source: news | social | press | web | all"),
) -> Dict[str, Any]:
    """
    Perform sentiment analysis for a given equity ticker.

    Args:
        ticker (str): Equity ticker symbol.
        source (str): Feed source (default 'all').

    Returns:
        Dict[str, Any]: Aggregated sentiment results.
    """
    try:
        # Use provided ticker or fallback to globally active equity
        equity_ticker = ticker or get_active_equity()
        logger.info(f"Sentiment request received: ticker={equity_ticker}, source={source}")

        # Instantiate aggregator (pulls feeds, preprocesses, extracts, runs sentiment model)
        aggregator = SentimentAggregator(ticker=equity_ticker)
        sentiment_results = aggregator.aggregate_sentiment(source=source)

        logger.info(f"Sentiment generated successfully for {equity_ticker}")
        return sentiment_results

    except ValueError as ve:
        logger.error(f"Active equity not set and ticker not provided: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {ticker or 'GLOBAL_EQUITY'}: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")