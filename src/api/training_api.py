"""
Training API router.

This module exposes endpoints for triggering model training jobs.
It connects the API layer with the training pipeline, keeping business
logic separate from request handling.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Dict

# Business logic import (training runner handles actual workflow)
from src.training.train_runner import TrainingRunner

# ------------------------------------------------------------
# Router & Logger Setup
# ------------------------------------------------------------
router = APIRouter()
logger = logging.getLogger("training_api")


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@router.post("/train", response_model=Dict[str, str])
async def train_model(
    ticker: str = Query(..., description="Equity ticker symbol (e.g., RELIANCE, AAPL)"),
    config_path: str = Query(..., description="Path to YAML config file for training"),
) -> Dict[str, str]:
    """
    Trigger model training for a given equity ticker using the specified config.

    Args:
        ticker (str): Equity ticker symbol.
        config_path (str): Path to YAML configuration file.

    Returns:
        Dict[str, str]: Training job status message.
    """
    try:
        logger.info(f"Training request received: ticker={ticker}, config={config_path}")

        runner = TrainingRunner(ticker=ticker, config_path=config_path)
        job_id = runner.run_training()

        logger.info(f"Training started successfully for {ticker} (job_id={job_id})")
        return {"status": "started", "job_id": job_id}

    except Exception as e:
        logger.error(f"Training failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Training job failed")
