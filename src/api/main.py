"""
FastAPI application entrypoint for the Equity Forecasting API.

This module defines the FastAPI app, basic endpoints, and 
a function to start the server via Uvicorn.

Usage:
    python src/api/main.py
"""

from fastapi import FastAPI
import uvicorn
from src.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Create FastAPI application instance with title metadata
app = FastAPI(title="Equity Forecasting API")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the API server is running.

    Returns:
        dict: A simple JSON response with status 'ok'
    """
    logger.info("Health check requested")
    return {"status": "ok"}


def start_api(host: str = "0.0.0.0", port: int = 8000):
    """
    Starts the FastAPI server using Uvicorn.

    Args:
        host (str): Host address to bind (default '0.0.0.0' to listen on all interfaces).
        port (int): Port number to listen on (default 8000).
    """
    logger.info(f"Starting API server on {host}:{port}")
    # Run Uvicorn server with reload enabled for development
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)


# Run server if executed as main script
if __name__ == "__main__":
    start_api()
