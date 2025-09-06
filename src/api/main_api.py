"""
Main-API entrypoint for the Equity Forecasting project.

This module initializes the FastAPI app, configures middleware for
logging and error handling, and registers sub-routers for forecasting,
sentiment analysis, and training functionalities.

Usage:
    uvicorn src.api.main_api:app --reload --port 8000
"""

import logging
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

# Import routers (modular APIs for different domains)
from src.api.forecasting_api import router as forecasting_router
from src.api.sentiment_api import router as sentiment_router
from src.api.training_api import router as training_router


# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("main_api")


# ------------------------------------------------------------
# FastAPI App Initialization
# ------------------------------------------------------------
app = FastAPI(
    title="Equity Forecasting API",
    description="API layer for equity price forecasting and sentiment analysis.",
    version="1.0.0",
)


# ------------------------------------------------------------
# Middleware
# ------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log each incoming request and its response time.
    """
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url}")

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    process_time = time.time() - start_time
    logger.info(
        f"Completed request: {request.method} {request.url} "
        f"Status: {response.status_code} Time: {process_time:.2f}s"
    )
    return response


# Allow frontend/Dashboard to access APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be restricted later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------------
@app.get("/health", response_model=Dict[str, str])
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for monitoring systems.
    """
    return {
        "status": "ok",
        "uptime": "active",
        "timestamp": str(time.time()),
    }


# ------------------------------------------------------------
# Register Routers
# ------------------------------------------------------------
app.include_router(forecasting_router, prefix="/forecast", tags=["Forecasting"])
app.include_router(sentiment_router, prefix="/sentiment", tags=["Sentiment"])
app.include_router(training_router, prefix="/train", tags=["Training"])
