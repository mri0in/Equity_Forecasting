import logging
import io
import pytest
from pathlib import Path
from src.utils.logger import get_logger

# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def tmp_log_file(tmp_path):
    """Temporary log file path."""
    return tmp_path / "test.log"


@pytest.fixture
def capture_logger_stream(monkeypatch):
    """Patch sys.stdout to StringIO for capturing logs."""
    buffer = io.StringIO()
    monkeypatch.setattr("sys.stdout", buffer)
    return buffer


# -----------------------
# Tests
# -----------------------
def test_console_logging(capture_logger_stream):
    """Logger prints to console."""
    logger = get_logger("test_console", level="INFO")
    logger.info("console log message")
    # Flush handlers
    for handler in logger.handlers:
        handler.flush()
    content = capture_logger_stream.getvalue()
    assert "test_console" in content
    assert "console log message" in content


def test_file_logging(tmp_log_file):
    """Logger writes to a file."""
    log_file_path = str(tmp_log_file)
    logger = get_logger(f"test_file_{tmp_log_file.name}", level="DEBUG", log_file=log_file_path)
    logger.error("file log message")
    # Flush all handlers
    for handler in logger.handlers:
        handler.flush()
    # File should exist and contain message
    assert tmp_log_file.exists(), f"Log file not created at {tmp_log_file}"
    with open(tmp_log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "file log message" in content


def test_logger_level_debug(capture_logger_stream):
    """Logger respects DEBUG level."""
    logger = get_logger("test_debug", level="DEBUG")
    logger.debug("debug message")
    for h in logger.handlers:
        h.flush()
    assert "debug message" in capture_logger_stream.getvalue()


def test_logger_level_warning(capture_logger_stream):
    """Logger respects WARNING level."""
    logger = get_logger("test_warning", level="WARNING")
    logger.warning("warning message")
    logger.info("info message")  # should not appear
    for h in logger.handlers:
        h.flush()
    content = capture_logger_stream.getvalue()
    assert "warning message" in content
    assert "info message" not in content


def test_logger_reuse(tmp_path):
    """Repeated get_logger calls do not duplicate handlers."""
    log_file_path = tmp_path / "reuse.log"
    logger1 = get_logger("reuse_logger", log_file=str(log_file_path))
    logger2 = get_logger("reuse_logger", log_file=str(log_file_path))
    assert logger1 is logger2
    # Only two handlers: console + file
    assert len(logger1.handlers) == 2


def test_file_logging_directory_creation(tmp_path):
    """Logger creates intermediate directories if missing."""
    nested_dir = tmp_path / "nested" / "dir"
    log_file_path = nested_dir / "nested.log"
    logger = get_logger("nested_logger", log_file=str(log_file_path))
    logger.error("nested log message")
    for h in logger.handlers:
        h.flush()
    assert log_file_path.exists()
    with open(log_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "nested log message" in content
