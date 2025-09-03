"""
Unit tests for CLI dispatcher in main.py

These tests verify that the correct orchestrator methods are triggered
when running CLI commands like train, predict, optimize, ensemble,
walkforward, and pipeline.
"""

import pytest
from unittest.mock import patch, MagicMock
from main import main


def run_cli_and_capture(monkeypatch, args, orchestrator_mock):
    """
    Helper to run CLI with patched orchestrator and sys.argv.
    """
    with patch("main.PipelineOrchestrator", return_value=orchestrator_mock):
        monkeypatch.setattr("sys.argv", ["main.py"] + args)
        main()


def test_pipeline_command_triggers_orchestrator(monkeypatch):
    """Ensure 'pipeline' command calls orchestrator.run_pipeline()."""
    mock_orchestrator = MagicMock()
    run_cli_and_capture(monkeypatch, ["pipeline", "--config", "src/config/config.yaml"], mock_orchestrator)
    mock_orchestrator.run_pipeline.assert_called_once_with()


def test_train_command_triggers_orchestrator(monkeypatch):
    """Ensure 'train' command calls orchestrator.run_pipeline(['train'])."""
    mock_orchestrator = MagicMock()
    run_cli_and_capture(monkeypatch, ["train", "--config", "src/config/config.yaml"], mock_orchestrator)
    mock_orchestrator.run_pipeline.assert_called_once_with(["train"])


def test_predict_command_triggers_orchestrator(monkeypatch):
    """Ensure 'predict' command calls orchestrator.run_pipeline(['predict'])."""
    mock_orchestrator = MagicMock()
    run_cli_and_capture(monkeypatch, ["predict", "--config", "src/config/config.yaml"], mock_orchestrator)
    mock_orchestrator.run_pipeline.assert_called_once_with(["predict"])


def test_optimize_command_triggers_orchestrator(monkeypatch):
    """Ensure 'optimize' command calls orchestrator.run_pipeline(['optimize'])."""
    mock_orchestrator = MagicMock()
    run_cli_and_capture(monkeypatch, ["optimize", "--config", "src/config/config.yaml"], mock_orchestrator)
    mock_orchestrator.run_pipeline.assert_called_once_with(["optimize"])


def test_ensemble_command_triggers_orchestrator(monkeypatch):
    """Ensure 'ensemble' command calls orchestrator.run_pipeline(['ensemble'])."""
    mock_orchestrator = MagicMock()
    run_cli_and_capture(monkeypatch, ["ensemble", "--config", "src/config/config.yaml"], mock_orchestrator)
    mock_orchestrator.run_pipeline.assert_called_once_with(["ensemble"])


def test_walkforward_command_triggers_orchestrator(monkeypatch):
    """Ensure 'walkforward' command calls orchestrator.run_pipeline(['walkforward'])."""
    mock_orchestrator = MagicMock()
    run_cli_and_capture(monkeypatch, ["walkforward", "--config", "src/config/config.yaml"], mock_orchestrator)
    mock_orchestrator.run_pipeline.assert_called_once_with(["walkforward"])
