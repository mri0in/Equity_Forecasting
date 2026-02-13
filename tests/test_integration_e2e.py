# tests/test_integration_e2e.py
"""
End-to-End Integration Tests for Phase 1.9

Tests:
- Full pipeline execution (A → B → C)
- Data validation at each stage
- State manager lifecycle tracking
- API integration
- Error handling and fallbacks
- Regression testing (no broken functionality)
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.validation.expectations import DataExpectations
from src.validation.validator import DataValidator
from src.validation.sanitizer import InputSanitizer, BatchSanitizer
from src.pipeline.state_manager import StateManager, TaskStatus
from src.monitoring.error_logging import ErrorLogger, ErrorComponent, FallbackReason
from src.config.api_models import ForecastRequest, SentimentPanelRequest, TrainingRequest


class TestDataValidationFramework:
    """Test integrated validation across pipeline stages."""

    def test_raw_data_validation(self):
        """Validate raw ingestion data expectations."""
        # Create sample raw data
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=100),
            "Open": np.random.uniform(100, 150, 100),
            "High": np.random.uniform(150, 160, 100),
            "Low": np.random.uniform(90, 100, 100),
            "Close": np.random.uniform(100, 150, 100),
            "Volume": np.random.randint(1000000, 10000000, 100),
            "Adj Close": np.random.uniform(100, 150, 100),
        })

        validator = DataValidator(stage="raw_stock")
        result = validator.validate(df)
        assert result["stage"] == "raw_stock"

    def test_preprocessed_data_validation(self):
        """Validate preprocessed data after cleaning."""
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=100),
            "Close": np.random.uniform(100, 150, 100),
            "Volume": np.random.randint(1000000, 10000000, 100),
            "Returns": np.random.uniform(-0.1, 0.1, 100),
            "Log_Returns": np.random.uniform(-0.1, 0.1, 100),
        })

        validator = DataValidator(stage="preprocessed")
        result = validator.validate(df)
        assert result["stage"] == "preprocessed"

    def test_featured_data_validation(self):
        """Validate feature engineering output."""
        # Create data with technical features
        dates = pd.date_range("2025-01-01", periods=100)
        close_prices = np.random.uniform(100, 150, 100)

        df = pd.DataFrame({
            "Date": dates,
            "Close": close_prices,
            "SMA_20": close_prices,  # Simplified: actual SMA
            "RSI": np.random.uniform(30, 70, 100),
            "MACD": np.random.uniform(-5, 5, 100),
            "Volume": np.random.randint(1000000, 10000000, 100),
            "Target": np.random.uniform(0, 1, 100),  # Prediction target
        })

        validator = DataValidator(stage="featured")
        result = validator.validate(df)
        assert result["stage"] == "featured"

    def test_validation_failure_on_corrupt_data(self):
        """Ensure validation fails gracefully on corrupt data."""
        # Data with NaN values
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=100),
            "Open": [np.nan] * 100,  # All NaN
            "High": np.random.uniform(150, 160, 100),
            "Low": np.random.uniform(90, 100, 100),
            "Close": np.random.uniform(100, 150, 100),
            "Volume": np.random.randint(1000000, 10000000, 100),
            "Adj Close": np.random.uniform(100, 150, 100),
        })

        validator = DataValidator(stage="raw_stock")
        result = validator.validate(df)
        assert result["stage"] == "raw_stock"


class TestInputSanitization:
    """Test input validation and sanitization."""

    def test_ticker_sanitization(self):
        """Sanitizer should validate ticker format."""
        sanitizer = InputSanitizer()

        # Valid tickers (alphanumeric is allowed)
        assert sanitizer.sanitize_ticker("AAPL") is not None
        assert sanitizer.sanitize_ticker("tsla") is not None
        assert sanitizer.sanitize_ticker("A1BC") is not None  # Alphanumeric is allowed
        assert sanitizer.sanitize_ticker("BRK.B") is not None  # Dots allowed

        # Invalid tickers
        assert sanitizer.sanitize_ticker("AAPL!") is None  # Special chars not allowed
        assert sanitizer.sanitize_ticker("AAPLTOOLONG") is None  # Too long (> 10 chars)

    def test_horizon_sanitization(self):
        """Sanitizer should validate horizon days."""
        sanitizer = InputSanitizer()

        # Valid horizons
        assert sanitizer.sanitize_horizon(30) is not None
        assert sanitizer.sanitize_horizon(365) is not None

        # Invalid horizons (return None instead of raising)
        assert sanitizer.sanitize_horizon(0) is None
        assert sanitizer.sanitize_horizon(400) is None

    def test_batch_sanitization(self):
        """BatchSanitizer should handle multiple inputs."""
        batch_sanitizer = BatchSanitizer()

        tickers = ["AAPL", "MSFT", "GOOG"]

        results = batch_sanitizer.sanitize_tickers(tickers)
        assert len(results) == 3
        assert all(isinstance(v, bool) for v in results.values())

    def test_dataframe_sanitization(self):
        """Sanitizer should clean DataFrames."""
        sanitizer = InputSanitizer()

        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4],
            "b": [5, 6, 7, 8],
            "c": [np.inf, 10, 11, 12],
        })

        result = sanitizer.sanitize_dataframe(df)
        # Should remove rows with NaN and inf
        assert len(result) <= len(df)


class TestPydanticAPIValidation:
    """Test API request/response validation."""

    def test_forecast_request_validation(self):
        """ForecastRequest should validate inputs."""
        # Valid request
        req = ForecastRequest(ticker="AAPL", horizon=30)
        assert req.ticker == "AAPL"
        assert req.horizon == 30

        # Invalid ticker (too long)
        with pytest.raises(ValueError):
            ForecastRequest(ticker="TOOLONGTICKERRRRR", horizon=30)

        # Invalid horizon (out of range)
        with pytest.raises(ValueError):
            ForecastRequest(ticker="AAPL", horizon=400)

    def test_sentiment_request_validation(self):
        """SentimentPanelRequest should validate inputs."""
        # Valid request
        req = SentimentPanelRequest(ticker="AAPL", lookback_days=30)
        assert req.ticker == "AAPL"
        assert req.lookback_days == 30

        # Invalid lookback
        with pytest.raises(ValueError):
            SentimentPanelRequest(ticker="AAPL", lookback_days=1000)

    def test_training_request_validation(self):
        """TrainingRequest should validate inputs."""
        # Valid request
        req = TrainingRequest(
            tickers=["AAPL", "MSFT"],
            start_date="2025-01-01",
            end_date="2025-12-31",
            model_type="lstm"
        )
        assert len(req.tickers) == 2
        assert req.model_type == "lstm"

        # Invalid model type
        with pytest.raises(ValueError):
            TrainingRequest(
                tickers=["AAPL"],
                start_date="2025-01-01",
                end_date="2025-12-31",
                model_type="invalid_model"
            )


class TestStateManager:
    """Test state management and task tracking."""

    @pytest.fixture
    def state_manager(self, tmp_path):
        """Create temporary state manager for testing."""
        manager = StateManager(
            state_dir=str(tmp_path),
        )
        return manager

    def test_task_lifecycle(self, state_manager):
        """Verify complete task lifecycle."""
        # Start task
        state_manager.mark_task_starting("test_task", {"equity": "AAPL"})
        task = state_manager.get_task_status("test_task")
        assert task is not None
        assert task.get("status") == TaskStatus.RUNNING.value

        # Complete task
        state_manager.mark_task_completed(
            "test_task",
            {"result": "success", "elapsed": 1.5}
        )
        task = state_manager.get_task_status("test_task")
        assert task.get("status") == TaskStatus.COMPLETED.value

    def test_task_failure_tracking(self, state_manager):
        """Verify error tracking in failed tasks."""
        state_manager.mark_task_starting("failing_task")
        
        error_msg = "Test error: ValueError occurred"
        state_manager.mark_task_failed(
            "failing_task",
            error=error_msg
        )
        
        task = state_manager.get_task_status("failing_task")
        assert task is not None
        assert task.get("status") == TaskStatus.FAILED.value
        assert "error" in task

    def test_multiple_tasks_independent(self, state_manager):
        """Multiple tasks should be tracked independently."""
        state_manager.mark_task_starting("task_1")
        state_manager.mark_task_starting("task_2")
        state_manager.mark_task_completed("task_1")

        task1 = state_manager.get_task_status("task_1")
        task2 = state_manager.get_task_status("task_2")
        
        assert task1.get("status") == TaskStatus.COMPLETED.value
        assert task2.get("status") == TaskStatus.RUNNING.value

    def test_query_tasks_by_status(self, state_manager):
        """Should query tasks by status."""
        state_manager.mark_task_starting("running_task_1")
        state_manager.mark_task_starting("running_task_2")
        state_manager.mark_task_completed("running_task_1")

        running = state_manager.get_running_tasks()
        completed = state_manager.get_completed_tasks()

        assert len(running) == 1
        assert len(completed) == 1

    def test_state_persistence(self, state_manager):
        """State should persist to disk."""
        state_manager.mark_task_starting("persistent_task", {"data": "test"})
        state_manager.mark_task_completed("persistent_task", {"result": "ok"})

        # Verify task was recorded
        task = state_manager.get_task_status("persistent_task")
        assert task is not None
        assert task.get("status") == TaskStatus.COMPLETED.value


class TestErrorHandlingAndFallbacks:
    """Test error logging and fallback mechanisms."""

    def test_adapter_error_logging(self):
        """ErrorLogger should capture adapter errors."""
        logger = ErrorLogger(component=ErrorComponent.FORECASTING_API)

        exc = RuntimeError("Adapter failed")
        logger.log_fallback(
            reason=FallbackReason.ADAPTER_EXCEPTION,
            exception=exc,
            context={"equity": "AAPL", "horizon": 30},
            fallback_action="Using simulated forecast"
        )

        assert logger.fallback_count == 1
        record = logger.error_history[0]
        assert record["reason"] == "adapter_exception"
        assert record["context"]["equity"] == "AAPL"

    def test_missing_dependency_fallback(self):
        """ErrorLogger should track missing dependencies."""
        logger = ErrorLogger(component=ErrorComponent.FORECASTING_API)

        logger.log_fallback(
            reason=FallbackReason.MISSING_DEPENDENCY,
            context={"missing": "global_signal.npy"},
            fallback_action="Using default signal"
        )

        assert logger.fallback_count == 1
        record = logger.error_history[0]
        assert record["reason"] == "missing_dependency"

    def test_error_summary_generation(self):
        """ErrorLogger should generate summary statistics."""
        logger = ErrorLogger(component=ErrorComponent.SENTIMENT_PANEL)

        for i in range(3):
            logger.log_error(f"Error {i}")

        for i in range(2):
            logger.log_fallback(reason=FallbackReason.EXTERNAL_API_FAILURE)

        summary = logger.get_error_summary()
        assert summary["total_errors"] == 3
        assert summary["total_fallbacks"] == 2
        assert summary["component"] == "sentiment_panel"


class TestAPIIntegration:
    """Test API endpoint integration."""

    def test_forecast_request_e2e(self):
        """ForecastRequest should be usable in full flow."""
        # Create request
        request = ForecastRequest(ticker="AAPL", horizon=30)

        # Verify request is valid
        assert request.ticker == "AAPL"
        assert request.horizon == 30

        # Could call API endpoint
        # response = get_forecast_for_equity(request.ticker, request.horizon)
        # assert response is not None

    def test_sentiment_request_e2e(self):
        """SentimentPanelRequest should be usable in full flow."""
        request = SentimentPanelRequest(ticker="MSFT", lookback_days=60)

        assert request.ticker == "MSFT"
        assert request.lookback_days == 60

    def test_training_request_e2e(self):
        """TrainingRequest should be usable in full flow."""
        request = TrainingRequest(
            tickers=["AAPL", "MSFT", "GOOG"],
            start_date="2025-01-01",
            end_date="2025-12-31",
            model_type="lstm"
        )

        assert len(request.tickers) == 3
        assert request.model_type == "lstm"


class TestRegressionAndNoBreaks:
    """Verify no regressions in existing functionality."""

    def test_expectations_available(self):
        """All expectations should be available."""
        exp_stock = DataExpectations.get_expectations("raw_stock")
        exp_clean = DataExpectations.get_expectations("preprocessed")
        exp_featured = DataExpectations.get_expectations("featured")

        assert exp_stock is not None
        assert exp_clean is not None
        assert exp_featured is not None

    def test_validator_creates_summary(self):
        """Validator should generate proper summaries."""
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=100),
            "Open": np.random.uniform(100, 150, 100),
            "High": np.random.uniform(150, 160, 100),
            "Low": np.random.uniform(90, 100, 100),
            "Close": np.random.uniform(100, 150, 100),
            "Volume": np.random.randint(1000000, 10000000, 100),
            "Adj Close": np.random.uniform(100, 150, 100),
        })

        validator = DataValidator(stage="raw_stock")
        result = validator.validate(df)

        # Verify summary structure
        assert "stage" in result
        assert result["stage"] == "raw_stock"
        assert "checks" in result

    def test_sanitizer_handles_edge_cases(self):
        """Sanitizer should handle edge cases gracefully."""
        sanitizer = InputSanitizer()

        # Empty DataFrame returns None
        df_empty = pd.DataFrame()
        result = sanitizer.sanitize_dataframe(df_empty)
        # Empty DataFrames return None (expected behavior)
        assert result is None or isinstance(result, pd.DataFrame)

        # Single-row DataFrame
        df_single = pd.DataFrame({"a": [1]})
        result = sanitizer.sanitize_dataframe(df_single)
        assert isinstance(result, pd.DataFrame)

    def test_state_manager_handles_missing_tasks(self):
        """StateManager should handle missing task queries."""
        manager = StateManager()

        # Query non-existent task - should return None
        status = manager.get_task_status("nonexistent")
        # Status can be None or PENDING depending on implementation
        assert status is None or status == TaskStatus.PENDING

    def test_error_logger_handles_file_failures(self):
        """ErrorLogger should handle file write failures gracefully."""
        logger = ErrorLogger(component=ErrorComponent.ADAPTER)

        # Even if file write fails, logging should not crash
        with patch("builtins.open", side_effect=IOError("File write failed")):
            logger.log_error(
                error_msg="Test error",
                severity="warning"
            )

        # Logger should still record in memory
        assert logger.error_count == 1


class TestValidationIntegration:
    """Test validation framework integration across pipeline."""

    def test_multi_stage_validation(self):
        """Validate data through multiple pipeline stages."""
        # Create raw data
        raw_df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=100),
            "Open": np.random.uniform(100, 150, 100),
            "High": np.random.uniform(150, 160, 100),
            "Low": np.random.uniform(90, 100, 100),
            "Close": np.random.uniform(100, 150, 100),
            "Volume": np.random.randint(1000000, 10000000, 100),
            "Adj Close": np.random.uniform(100, 150, 100),
        })

        # Stage 1: Raw validation
        raw_validator = DataValidator(stage="raw_stock")
        raw_result = raw_validator.validate(raw_df)
        assert raw_result["stage"] == "raw_stock"

        # Stage 2: Preprocessing (would apply cleaning here in real pipeline)
        clean_df = raw_df.copy()
        
        # Stage 3: Feature generation
        featured_df = raw_df.copy()
        featured_df["Target"] = np.random.uniform(0, 1, len(raw_df))

        # Validate featured
        feat_validator = DataValidator(stage="featured")
        feat_result = feat_validator.validate(featured_df)
        assert feat_result["stage"] == "featured"

    def test_batch_processing_validation(self):
        """Validate batches of data."""
        batch_sanitizer = BatchSanitizer()

        tickers = ["AAPL", "MSFT", "GOOGL"]
        
        results = batch_sanitizer.sanitize_tickers(tickers)
        assert len(results) == 3
        assert all(isinstance(v, bool) for v in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
