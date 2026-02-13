# src/monitoring/error_logging.py
"""
Error Logging Framework for Equity Forecasting.

Replaces silent fallbacks with explicit logging, fallback reasons, and component tagging.
Provides structured error tracking and monitoring across all components.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone
from pathlib import Path
import json
from enum import Enum


class ErrorComponent(Enum):
    """Component identifiers for error tracking and monitoring."""
    SENTIMENT_PANEL = "sentiment_panel"
    FORECASTING_API = "forecasting_api"
    ADAPTER = "adapter"
    DATA_PROCESSOR = "data_processor"
    ENSEMBLE = "ensemble"
    LSTM_MODEL = "lstm_model"
    TRAINING = "training"
    SENTIMENT_AGGREGATOR = "sentiment_aggregator"
    DATA_VALIDATION = "data_validation"
    INPUT_SANITIZER = "input_sanitizer"
    STATE_MANAGER = "state_manager"


class FallbackReason(Enum):
    """Reasons why fallback was triggered."""
    ADAPTER_EXCEPTION = "adapter_exception"
    MISSING_DEPENDENCY = "missing_dependency"
    CORRUPT_DATA = "corrupt_data"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_INPUT = "invalid_input"
    INSUFFICIENT_DATA = "insufficient_data"
    EXTERNAL_API_FAILURE = "external_api_failure"
    UNKNOWN = "unknown"


class ErrorLogger:
    """
    Structured error logging with component tagging and fallback tracking.
    
    Usage:
        logger = ErrorLogger(component=ErrorComponent.FORECASTING_API)
        try:
            result = risky_operation()
        except Exception as exc:
            logger.log_fallback(
                reason=FallbackReason.ADAPTER_EXCEPTION,
                exception=exc,
                context={"equity": "AAPL", "horizon": 30}
            )
            result = fallback_operation()
    """
    
    def __init__(self, component: ErrorComponent, base_logger: Optional[logging.Logger] = None):
        """
        Initialize error logger for a specific component.
        
        Args:
            component: ErrorComponent enum identifying the component
            base_logger: Optional logging.Logger to use (creates default if None)
        """
        self.component = component
        self.logger = base_logger or logging.getLogger(f"error.{component.value}")
        self.logger.setLevel(logging.INFO)
        
        # Error statistics
        self.error_count = 0
        self.fallback_count = 0
        self.error_history: list[Dict[str, Any]] = []
        
        # Error log file
        log_dir = Path(__file__).resolve().parent.parent.parent / "datalake" / "runs"
        self.error_log_path = log_dir / "error_log.jsonl"
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_fallback(
        self,
        reason: FallbackReason,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        fallback_action: Optional[str] = None,
    ) -> None:
        """
        Log an error event with fallback information.
        
        Args:
            reason: FallbackReason enum indicating why fallback occurred
            exception: Optional exception that triggered the fallback
            context: Optional context dict (equity, horizon, model type, etc.)
            fallback_action: Optional description of fallback action taken
        """
        self.fallback_count += 1
        
        error_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": self.component.value,
            "reason": reason.value,
            "fallback_count": self.fallback_count,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_message": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception else None,
            "context": context or {},
            "fallback_action": fallback_action or "Using fallback simulation",
        }
        
        self.error_history.append(error_record)
        
        # Format log message
        context_str = ", ".join(f"{k}={v}" for k, v in (context or {}).items())
        exc_str = f": {exception}" if exception else ""
        
        log_msg = (
            f"[{self.component.value.upper()}] "
            f"Fallback triggered ({reason.value}){exc_str} "
            f"| Context: {context_str} "
            f"| Action: {fallback_action or 'Using simulated data'}"
        )
        
        self.logger.warning(log_msg)
        
        # Persist to error log
        self._persist_error(error_record)
    
    def log_error(
        self,
        error_msg: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "warning",
    ) -> None:
        """
        Log a general error (not necessarily triggering fallback).
        
        Args:
            error_msg: Description of the error
            exception: Optional exception object
            context: Optional context dict
            severity: 'debug', 'info', 'warning', 'error', 'critical'
        """
        self.error_count += 1
        
        error_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": self.component.value,
            "message": error_msg,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_message": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception else None,
            "context": context or {},
            "severity": severity,
        }
        
        self.error_history.append(error_record)
        
        # Log to standard logger
        log_func = getattr(self.logger, severity, self.logger.warning)
        context_str = ", ".join(f"{k}={v}" for k, v in (context or {}).items())
        log_msg = f"[{self.component.value.upper()}] {error_msg} | Context: {context_str}"
        log_func(log_msg)
        
        # Persist to error log
        self._persist_error(error_record)
    
    def _persist_error(self, error_record: Dict[str, Any]) -> None:
        """Append error record to JSONL error log file."""
        try:
            with open(self.error_log_path, "a") as f:
                f.write(json.dumps(error_record) + "\n")
        except Exception as e:
            # Fallback to stderr if file write fails
            self.logger.error(f"Failed to write error log: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary statistics of errors logged by this component."""
        return {
            "component": self.component.value,
            "total_errors": self.error_count,
            "total_fallbacks": self.fallback_count,
            "recent_errors": self.error_history[-10:] if self.error_history else [],
        }
    
    def clear_history(self) -> None:
        """Clear in-memory error history."""
        self.error_history.clear()


def wrap_with_fallback(
    fallback_func: Callable,
    component: ErrorComponent,
    reason: FallbackReason = FallbackReason.UNKNOWN,
    context: Optional[Dict[str, Any]] = None,
    fallback_action: Optional[str] = None,
) -> Callable:
    """
    Decorator to wrap a function with error handling and fallback.
    
    Usage:
        @wrap_with_fallback(
            fallback_func=lambda: {"result": "simulated"},
            component=ErrorComponent.FORECASTING_API,
            reason=FallbackReason.ADAPTER_EXCEPTION,
        )
        def get_forecast(equity: str) -> Dict:
            # ... actual implementation ...
    
    Args:
        fallback_func: Function to call if main function raises exception
        component: ErrorComponent for this operation
        reason: Primary FallbackReason expected
        context: Optional context dict to log
        fallback_action: Optional description of fallback action
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            error_logger = ErrorLogger(component=component)
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                error_logger.log_fallback(
                    reason=reason,
                    exception=exc,
                    context=context,
                    fallback_action=fallback_action,
                )
                return fallback_func()
        return wrapper
    return decorator


# Factory function to create component-specific loggers
def create_component_logger(component: ErrorComponent) -> ErrorLogger:
    """Create a component-specific error logger."""
    return ErrorLogger(component=component)
