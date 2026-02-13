# tests/test_error_logging.py
"""
Tests for error logging framework (Phase 1.7).

Tests:
- ErrorLogger initialization and logging
- Component tagging
- Fallback reason tracking
- Error persistence to JSONL
- Decorator wrapping with fallback
- Error summary generation
"""

import pytest
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.monitoring.error_logging import (
    ErrorLogger,
    ErrorComponent,
    FallbackReason,
    create_component_logger,
    wrap_with_fallback,
)


class TestErrorComponent:
    """Test ErrorComponent enum values."""

    def test_sentiment_panel_component(self):
        """Sentiment panel should be registered."""
        assert ErrorComponent.SENTIMENT_PANEL.value == "sentiment_panel"

    def test_forecasting_api_component(self):
        """Forecasting API should be registered."""
        assert ErrorComponent.FORECASTING_API.value == "forecasting_api"

    def test_adapter_component(self):
        """Adapter should be registered."""
        assert ErrorComponent.ADAPTER.value == "adapter"


class TestFallbackReason:
    """Test FallbackReason enum values."""

    def test_adapter_exception_reason(self):
        """Adapter exception should be a valid reason."""
        assert FallbackReason.ADAPTER_EXCEPTION.value == "adapter_exception"

    def test_missing_dependency_reason(self):
        """Missing dependency should be a valid reason."""
        assert FallbackReason.MISSING_DEPENDENCY.value == "missing_dependency"

    def test_corrupt_data_reason(self):
        """Corrupt data should be a valid reason."""
        assert FallbackReason.CORRUPT_DATA.value == "corrupt_data"

    def test_external_api_failure_reason(self):
        """External API failure should be a valid reason."""
        assert FallbackReason.EXTERNAL_API_FAILURE.value == "external_api_failure"


class TestErrorLogger:
    """Test ErrorLogger class."""

    def test_initialization(self):
        """ErrorLogger should initialize with component."""
        logger = ErrorLogger(component=ErrorComponent.FORECASTING_API)
        assert logger.component == ErrorComponent.FORECASTING_API
        assert logger.error_count == 0
        assert logger.fallback_count == 0
        assert logger.error_history == []

    def test_log_fallback_with_exception(self):
        """log_fallback should record exception details."""
        logger = ErrorLogger(component=ErrorComponent.FORECASTING_API)
        
        exc = ValueError("Test error")
        logger.log_fallback(
            reason=FallbackReason.CORRUPT_DATA,
            exception=exc,
            context={"equity": "AAPL"},
            fallback_action="Using simulated data"
        )
        
        assert logger.fallback_count == 1
        assert len(logger.error_history) == 1
        
        record = logger.error_history[0]
        assert record["component"] == "forecasting_api"
        assert record["reason"] == "corrupt_data"
        assert record["exception_type"] == "ValueError"
        assert "Test error" in record["exception_message"]
        assert record["context"]["equity"] == "AAPL"
        assert record["fallback_action"] == "Using simulated data"

    def test_log_fallback_without_exception(self):
        """log_fallback should work without exception."""
        logger = ErrorLogger(component=ErrorComponent.SENTIMENT_PANEL)
        
        logger.log_fallback(
            reason=FallbackReason.EXTERNAL_API_FAILURE,
            context={"ticker": "GOOG"}
        )
        
        assert logger.fallback_count == 1
        record = logger.error_history[0]
        assert record["exception_type"] is None

    def test_log_error_without_exception(self):
        """log_error should record non-exception errors."""
        logger = ErrorLogger(component=ErrorComponent.ADAPTER)
        
        logger.log_error(
            error_msg="Configuration missing",
            context={"config_key": "model_path"},
            severity="warning"
        )
        
        assert logger.error_count == 1
        record = logger.error_history[0]
        assert record["message"] == "Configuration missing"
        assert record["severity"] == "warning"
        assert record["exception_type"] is None

    def test_log_error_with_exception(self):
        """log_error should capture exception details."""
        logger = ErrorLogger(component=ErrorComponent.ADAPTER)
        
        exc = RuntimeError("Critical failure")
        logger.log_error(
            error_msg="Runtime error occurred",
            exception=exc,
            severity="error"
        )
        
        assert logger.error_count == 1
        record = logger.error_history[0]
        assert record["exception_type"] == "RuntimeError"

    def test_get_error_summary(self):
        """get_error_summary should return statistics."""
        logger = ErrorLogger(component=ErrorComponent.FORECASTING_API)
        
        logger.log_error("Error 1")
        logger.log_fallback(reason=FallbackReason.ADAPTER_EXCEPTION)
        logger.log_error("Error 2")
        
        summary = logger.get_error_summary()
        assert summary["component"] == "forecasting_api"
        assert summary["total_errors"] == 2
        assert summary["total_fallbacks"] == 1
        assert len(summary["recent_errors"]) == 3

    def test_clear_history(self):
        """clear_history should empty error history."""
        logger = ErrorLogger(component=ErrorComponent.SENTIMENT_PANEL)
        
        logger.log_error("Error 1")
        assert len(logger.error_history) == 1
        
        logger.clear_history()
        assert len(logger.error_history) == 0

    def test_error_history_limited_to_10(self):
        """get_error_summary should return only last 10 errors."""
        logger = ErrorLogger(component=ErrorComponent.ADAPTER)
        
        # Log 15 errors
        for i in range(15):
            logger.log_error(f"Error {i}")
        
        summary = logger.get_error_summary()
        assert len(summary["recent_errors"]) == 10
        assert "Error 14" in summary["recent_errors"][-1]["message"]

    @patch("builtins.open", create=True)
    def test_persist_error_to_file(self, mock_open):
        """_persist_error should write to JSONL file."""
        logger = ErrorLogger(component=ErrorComponent.FORECASTING_API)
        
        record = {
            "timestamp": "2026-02-13T10:00:00",
            "component": "forecasting_api",
            "message": "Test error"
        }
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        logger._persist_error(record)
        
        # Verify file was opened in append mode
        mock_open.assert_called_once()
        call_args = mock_open.call_args
        assert call_args[0][1] == "a"  # Append mode
        
        # Verify JSON was written
        mock_file.write.assert_called_once()
        written_json = mock_file.write.call_args[0][0]
        assert json.loads(written_json)["component"] == "forecasting_api"

    def test_multiple_fallbacks_increment_counter(self):
        """Multiple fallbacks should increment counter correctly."""
        logger = ErrorLogger(component=ErrorComponent.SENTIMENT_PANEL)
        
        logger.log_fallback(reason=FallbackReason.EXTERNAL_API_FAILURE)
        assert logger.fallback_count == 1
        
        logger.log_fallback(reason=FallbackReason.CORRUPT_DATA)
        assert logger.fallback_count == 2
        
        record = logger.error_history[1]
        assert record["fallback_count"] == 2


class TestCreateComponentLogger:
    """Test create_component_logger factory function."""

    def test_creates_error_logger_with_component(self):
        """create_component_logger should return ErrorLogger."""
        logger = create_component_logger(ErrorComponent.ADAPTER)
        
        assert isinstance(logger, ErrorLogger)
        assert logger.component == ErrorComponent.ADAPTER

    def test_multiple_loggers_independent(self):
        """Multiple loggers should be independent."""
        logger1 = create_component_logger(ErrorComponent.FORECASTING_API)
        logger2 = create_component_logger(ErrorComponent.SENTIMENT_PANEL)
        
        logger1.log_error("Error 1")
        logger2.log_error("Error 2")
        
        assert logger1.error_count == 1
        assert logger2.error_count == 1
        assert len(logger1.error_history) == 1
        assert len(logger2.error_history) == 1


class TestWrapWithFallback:
    """Test wrap_with_fallback decorator."""

    def test_successful_function_execution(self):
        """Decorator should not interfere with successful execution."""
        
        def fallback():
            return {"result": "fallback"}
        
        @wrap_with_fallback(
            fallback_func=fallback,
            component=ErrorComponent.FORECASTING_API,
        )
        def risky_operation():
            return {"result": "success"}
        
        result = risky_operation()
        assert result["result"] == "success"

    def test_fallback_on_exception(self):
        """Decorator should call fallback on exception."""
        
        def fallback():
            return {"result": "fallback"}
        
        @wrap_with_fallback(
            fallback_func=fallback,
            component=ErrorComponent.FORECASTING_API,
            reason=FallbackReason.ADAPTER_EXCEPTION,
        )
        def risky_operation():
            raise ValueError("Operation failed")
        
        result = risky_operation()
        assert result["result"] == "fallback"

    def test_decorator_logs_exception(self):
        """Decorator should log exceptions."""
        
        def fallback():
            return {}
        
        exception_logged = []
        
        def mock_log_fallback(**kwargs):
            exception_logged.append(kwargs)
        
        @wrap_with_fallback(
            fallback_func=fallback,
            component=ErrorComponent.SENTIMENT_PANEL,
            reason=FallbackReason.EXTERNAL_API_FAILURE,
            context={"ticker": "AAPL"},
            fallback_action="Using simulated sentiment"
        )
        def risky_operation():
            raise RuntimeError("API call failed")
        
        with patch.object(ErrorLogger, "log_fallback", side_effect=mock_log_fallback):
            result = risky_operation()
        
        assert len(exception_logged) > 0

    def test_decorator_with_arguments(self):
        """Decorator should work with function arguments."""
        
        def fallback():
            return None
        
        @wrap_with_fallback(
            fallback_func=fallback,
            component=ErrorComponent.FORECASTING_API,
        )
        def operation_with_args(equity: str, horizon: int):
            if equity == "AAPL":
                return {"equity": equity, "horizon": horizon}
            raise ValueError("Unknown equity")
        
        # Successful call
        result = operation_with_args("AAPL", 30)
        assert result["equity"] == "AAPL"
        assert result["horizon"] == 30
        
        # Failed call with fallback
        result = operation_with_args("UNKNOWN", 30)
        assert result is None


class TestErrorLoggingIntegration:
    """Integration tests for error logging across components."""

    def test_sentiment_panel_error_logging(self):
        """Verify sentiment panel uses error logging."""
        from src.dashboard.sentiment_panel import error_logger
        
        assert error_logger.component == ErrorComponent.SENTIMENT_PANEL

    def test_forecasting_api_error_logging(self):
        """Verify forecasting API uses error logging."""
        from src.api.forecasting_api import error_logger
        
        assert error_logger.component == ErrorComponent.FORECASTING_API

    def test_component_logger_persistence(self):
        """Component loggers should write to error_log.jsonl."""
        logger = ErrorLogger(component=ErrorComponent.ADAPTER)
        
        # Record expected path
        expected_path = (
            Path(__file__).resolve().parent.parent / 
            "datalake" / "runs" / "error_log.jsonl"
        )
        
        assert logger.error_log_path == expected_path

    def test_error_context_captured(self):
        """Error context should be captured in records."""
        logger = ErrorLogger(component=ErrorComponent.FORECASTING_API)
        
        logger.log_error(
            error_msg="Stock data unavailable",
            context={
                "equity": "TSLA",
                "date_range": "2025-01",
                "source": "yfinance"
            }
        )
        
        record = logger.error_history[0]
        assert record["context"]["equity"] == "TSLA"
        assert record["context"]["date_range"] == "2025-01"
        assert record["context"]["source"] == "yfinance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
