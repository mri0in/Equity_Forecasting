"""Tests for API validation models."""

import pytest
from pydantic import ValidationError
from src.config.api_models import (
    ForecastRequest,
    SentimentPanelRequest,
    TrainingRequest,
    ForecastResponse,
    ErrorResponse
)


class TestForecastRequest:
    """Test ForecastRequest validation."""
    
    def test_valid_request(self):
        """Test valid forecast request."""
        req = ForecastRequest(ticker="AAPL", horizon=30)
        assert req.ticker == "AAPL"
        assert req.horizon == 30
    
    def test_ticker_uppercase(self):
        """Test ticker is uppercased."""
        req = ForecastRequest(ticker="aapl")
        assert req.ticker == "AAPL"
    
    def test_ticker_with_dot(self):
        """Test ticker with dot notation."""
        req = ForecastRequest(ticker="BRK.B")
        assert req.ticker == "BRK.B"
    
    def test_invalid_ticker_length(self):
        """Test ticker too long."""
        with pytest.raises(ValidationError):
            ForecastRequest(ticker="THISISTOOLONGTICKERINTALLY")
    
    def test_invalid_ticker_empty(self):
        """Test empty ticker."""
        with pytest.raises(ValidationError):
            ForecastRequest(ticker="")
    
    def test_invalid_ticker_chars(self):
        """Test invalid characters in ticker."""
        with pytest.raises(ValidationError):
            ForecastRequest(ticker="AAPL@#$")
    
    def test_horizon_minimum(self):
        """Test minimum horizon."""
        req = ForecastRequest(ticker="AAPL", horizon=1)
        assert req.horizon == 1
    
    def test_horizon_maximum(self):
        """Test maximum horizon."""
        req = ForecastRequest(ticker="AAPL", horizon=365)
        assert req.horizon == 365
    
    def test_horizon_below_minimum(self):
        """Test horizon below minimum."""
        with pytest.raises(ValidationError):
            ForecastRequest(ticker="AAPL", horizon=0)
    
    def test_horizon_above_maximum(self):
        """Test horizon above maximum."""
        with pytest.raises(ValidationError):
            ForecastRequest(ticker="AAPL", horizon=400)
    
    def test_optional_fields(self):
        """Test optional fields."""
        req = ForecastRequest(
            ticker="AAPL",
            horizon=30,
            include_confidence=True,
            model_version="v1.0"
        )
        assert req.include_confidence is True
        assert req.model_version == "v1.0"


class TestSentimentPanelRequest:
    """Test SentimentPanelRequest validation."""
    
    def test_valid_request(self):
        """Test valid sentiment request."""
        req = SentimentPanelRequest(ticker="AAPL", lookback_days=30)
        assert req.ticker == "AAPL"
        assert req.lookback_days == 30
    
    def test_with_sources(self):
        """Test with sentiment sources."""
        req = SentimentPanelRequest(
            ticker="AAPL",
            sources=["news", "twitter"]
        )
        assert "news" in req.sources
        assert "twitter" in req.sources
    
    def test_invalid_source(self):
        """Test invalid sentiment source."""
        with pytest.raises(ValidationError):
            SentimentPanelRequest(
                ticker="AAPL",
                sources=["invalidsource"]
            )
    
    def test_lookback_range(self):
        """Test lookback days range."""
        req_min = SentimentPanelRequest(ticker="AAPL", lookback_days=1)
        assert req_min.lookback_days == 1
        
        req_max = SentimentPanelRequest(ticker="AAPL", lookback_days=365)
        assert req_max.lookback_days == 365


class TestTrainingRequest:
    """Test TrainingRequest validation."""
    
    def test_valid_request(self):
        """Test valid training request."""
        req = TrainingRequest(
            tickers=["AAPL", "MSFT"],
            start_date="2022-01-01"
        )
        assert len(req.tickers) == 2
        assert req.start_date == "2022-01-01"
    
    def test_date_format_validation(self):
        """Test date format validation."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                tickers=["AAPL"],
                start_date="01-01-2022"  # Wrong format
            )
    
    def test_model_type_validation(self):
        """Test model type validation."""
        req = TrainingRequest(
            tickers=["AAPL"],
            start_date="2022-01-01",
            model_type="lstm"
        )
        assert req.model_type == "lstm"
        
        with pytest.raises(ValidationError):
            TrainingRequest(
                tickers=["AAPL"],
                start_date="2022-01-01",
                model_type="invalidmodel"
            )
    
    def test_test_size_range(self):
        """Test size fraction range."""
        req_min = TrainingRequest(
            tickers=["AAPL"],
            start_date="2022-01-01",
            test_size=0.05
        )
        assert req_min.test_size == 0.05
        
        req_max = TrainingRequest(
            tickers=["AAPL"],
            start_date="2022-01-01",
            test_size=0.5
        )
        assert req_max.test_size == 0.5
        
        with pytest.raises(ValidationError):
            TrainingRequest(
                tickers=["AAPL"],
                start_date="2022-01-01",
                test_size=0.01  # Below 5%
            )
    
    def test_empty_tickers(self):
        """Test empty tickers list."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                tickers=[],
                start_date="2022-01-01"
            )


class TestForecastResponse:
    """Test ForecastResponse model."""
    
    def test_valid_response(self):
        """Test valid forecast response."""
        resp = ForecastResponse(
            ticker="AAPL",
            forecast_date="2024-01-20",
            horizon=30,
            predicted_price=185.50,
            timestamp="2024-01-20T10:30:00Z"
        )
        assert resp.ticker == "AAPL"
        assert resp.predicted_price == 185.50
        assert resp.status == "success"
    
    def test_with_confidence_intervals(self):
        """Test response with confidence intervals."""
        resp = ForecastResponse(
            ticker="AAPL",
            forecast_date="2024-01-20",
            horizon=30,
            predicted_price=185.50,
            confidence_lower=182.00,
            confidence_upper=189.00,
            timestamp="2024-01-20T10:30:00Z"
        )
        assert resp.confidence_lower == 182.00
        assert resp.confidence_upper == 189.00


class TestErrorResponse:
    """Test ErrorResponse model."""
    
    def test_valid_error_response(self):
        """Test valid error response."""
        err = ErrorResponse(
            error_code="INVALID_TICKER",
            message="Ticker validation failed",
            timestamp="2024-01-20T10:30:00Z"
        )
        assert err.status == "error"
        assert err.error_code == "INVALID_TICKER"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
