"""Pydantic models for API request validation."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


class ForecastRequest(BaseModel):
    """Request model for forecast endpoint."""
    
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g., AAPL, BRK.B)"
    )
    horizon: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to forecast (1-365)"
    )
    include_confidence: bool = Field(
        default=False,
        description="Include confidence intervals in response"
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Specific model version to use (optional)"
    )
    
    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        """Validate ticker format."""
        if not v or not isinstance(v, str):
            raise ValueError("Ticker must be a non-empty string")
        
        v = v.strip().upper()
        
        # Allow alphanumeric, dots, dashes
        allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
        if not all(c in allowed_chars for c in v):
            raise ValueError(f"Ticker contains invalid characters: {v}")
        
        return v
    
    class Config:
        """Pydantic config."""
        example = {
            "ticker": "AAPL",
            "horizon": 30,
            "include_confidence": True,
            "model_version": None
        }


class SentimentPanelRequest(BaseModel):
    """Request model for sentiment panel endpoint."""
    
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol"
    )
    lookback_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days for sentiment analysis (1-365)"
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="Sentiment sources to include (news, twitter, etc.)"
    )
    
    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        """Validate ticker format."""
        if not v or not isinstance(v, str):
            raise ValueError("Ticker must be a non-empty string")
        
        v = v.strip().upper()
        allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
        if not all(c in allowed_chars for c in v):
            raise ValueError(f"Ticker contains invalid characters: {v}")
        
        return v
    
    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v):
        """Validate sentiment sources."""
        if v is None:
            return None
        
        if not isinstance(v, list):
            raise ValueError("Sources must be a list")
        
        valid_sources = {"news", "twitter", "reddit", "stocktwits", "seeking_alpha"}
        invalid = set(v) - valid_sources
        
        if invalid:
            raise ValueError(f"Invalid sources: {invalid}. Valid: {valid_sources}")
        
        return v
    
    class Config:
        """Pydantic config."""
        example = {
            "ticker": "AAPL",
            "lookback_days": 30,
            "sources": ["news", "twitter"]
        }


class TrainingRequest(BaseModel):
    """Request model for model training endpoint."""
    
    tickers: List[str] = Field(
        ...,
        min_items=1,
        description="List of tickers to train on"
    )
    start_date: str = Field(
        ...,
        description="Training start date (YYYY-MM-DD format)"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Training end date (YYYY-MM-DD format)"
    )
    model_type: str = Field(
        default="lstm",
        description="Model type: lstm, lightgbm, elastic_net, tcn"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Test set fraction (5-50%)"
    )
    
    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v):
        """Validate ticker list."""
        if not isinstance(v, list):
            raise ValueError("Tickers must be a list")
        
        if not all(isinstance(t, str) for t in v):
            raise ValueError("All tickers must be strings")
        
        return v
    
    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def validate_dates(cls, v):
        """Validate date format."""
        if v is None:
            return None
        
        if not isinstance(v, str):
            raise ValueError("Dates must be strings in YYYY-MM-DD format")
        
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")
    
    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v):
        """Validate model type."""
        valid_types = {"lstm", "lightgbm", "elastic_net", "tcn", "ridge"}
        if v not in valid_types:
            raise ValueError(f"Invalid model_type: {v}. Valid: {valid_types}")
        return v
    
    class Config:
        """Pydantic config."""
        example = {
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2022-01-01",
            "end_date": "2024-01-01",
            "model_type": "lstm",
            "test_size": 0.2
        }


class ForecastResponse(BaseModel):
    """Response model for forecast endpoint."""
    
    ticker: str
    forecast_date: str
    horizon: int
    predicted_price: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    model_version: Optional[str] = None
    timestamp: str
    status: str = "success"
    
    class Config:
        """Pydantic config."""
        example = {
            "ticker": "AAPL",
            "forecast_date": "2024-01-20",
            "horizon": 30,
            "predicted_price": 185.50,
            "confidence_lower": 182.00,
            "confidence_upper": 189.00,
            "model_version": "v1.0",
            "timestamp": "2024-01-20T10:30:00Z",
            "status": "success"
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    status: str = "error"
    error_code: str
    message: str
    details: Optional[dict] = None
    timestamp: str
    
    class Config:
        """Pydantic config."""
        example = {
            "status": "error",
            "error_code": "INVALID_TICKER",
            "message": "Ticker validation failed",
            "details": {"ticker": "Invalid characters in ticker"},
            "timestamp": "2024-01-20T10:30:00Z"
        }
