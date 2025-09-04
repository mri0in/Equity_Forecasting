import logging
import os
import pytest
from utils.logger import get_logger


def test_console_logging(capsys):
    """Test that logger prints to console when no log_file is provided."""
    logger = get_logger("test_console_logger", level="INFO")
    logger.info("console log message")

    captured = capsys.readouterr()
    assert "console log message" in captured.out
    assert "test_console_logger" in captured.out


def test_file_logging(tmp_path):
    """Test that logger writes logs to a file when log_file is provided."""
    log_file = tmp_path / "test.log"
    logger = get_logger("test_file_logger", level="DEBUG", log_file=str(log_file))
    logger.error("file log message")

    # Flush handlers to force write
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists(), f"Log file was not created at {log_file}"
    content = log_file.read_text()
    assert "file log message" in content
    assert "test_file_logger" in content


def test_logger_level_debug(capsys):
    """Test that DEBUG logs are captured when level=DEBUG is set."""
    logger = get_logger("test_debug_logger", level="DEBUG")
    logger.debug("debugging now")

    captured = capsys.readouterr()
    assert "debugging now" in captured.out
    assert "DEBUG" in captured.out


def test_logger_level_info_suppresses_debug(capsys):
    """Test that DEBUG messages are suppressed when level=INFO is set."""
    logger = get_logger("test_info_logger", level="INFO")
    logger.debug("this should NOT appear")
    logger.info("this should appear")

    captured = capsys.readouterr()
    assert "this should appear" in captured.out
    assert "this should NOT appear" not in captured.out


def test_logger_reuse_does_not_duplicate_handlers(tmp_path):
    """Test that calling get_logger multiple times does not duplicate handlers."""
    log_file = tmp_path / "reuse.log"

    logger1 = get_logger("test_reuse_logger", level="INFO", log_file=str(log_file))
    logger2 = get_logger("test_reuse_logger", level="INFO", log_file=str(log_file))

    assert logger1 is logger2
    assert len(logger1.handlers) == len(set(logger1.handlers))


def test_invalid_log_level_defaults_to_info(capsys):
    """Test that invalid log level falls back to INFO."""
    logger = get_logger("test_invalid_level_logger", level="NOT_A_LEVEL")
    logger.info("info fallback message")

    captured = capsys.readouterr()
    assert "info fallback message" in captured.out
    assert "INFO" in captured.out
