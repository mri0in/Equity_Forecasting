import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.G_wfv_pipeline import run_walk_forward_validation


def test_run_walk_forward_validation_success(monkeypatch):
    # Fake config and results
    fake_config = {"some_param": 1, "early_stopping": {"patience": 5}}
    fake_results = {
        "summary": {"rmse": 0.1, "mae": 0.05},
        "details": {"split_1": {"rmse": 0.1}, "split_2": {"rmse": 0.1}},
    }

    # Patch load_config to return fake_config
    monkeypatch.setattr("src.pipeline.run_walk_forward.load_config", lambda path: fake_config)

    # Patch WalkForwardValidator
    mock_validator = MagicMock()
    mock_validator.run_validation.return_value = fake_results
    monkeypatch.setattr("src.pipeline.run_walk_forward.WalkForwardValidator", lambda **kwargs: mock_validator)

    # Call function
    results = run_walk_forward_validation("config.yaml")

    # Assertions
    assert results == fake_results
    mock_validator.run_validation.assert_called_once()


def test_run_walk_forward_validation_no_early_stopping(monkeypatch):
    # Fake config without early stopping
    fake_config = {"some_param": 1}
    fake_results = {"summary": {"rmse": 0.2}, "details": {}}

    monkeypatch.setattr("src.pipeline.run_walk_forward.load_config", lambda path: fake_config)
    mock_validator = MagicMock()
    mock_validator.run_validation.return_value = fake_results
    monkeypatch.setattr("src.pipeline.run_walk_forward.WalkForwardValidator", lambda **kwargs: mock_validator)

    results = run_walk_forward_validation("config.yaml")

    assert results == fake_results
    mock_validator.run_validation.assert_called_once()


def test_run_walk_forward_validation_logger_called(monkeypatch):
    # Check if logger.info is called for each metric in summary
    fake_config = {"some_param": 1}
    fake_results = {"summary": {"rmse": 0.1, "mae": 0.05}, "details": {}}

    monkeypatch.setattr("src.pipeline.run_walk_forward.load_config", lambda path: fake_config)
    mock_validator = MagicMock()
    mock_validator.run_validation.return_value = fake_results
    monkeypatch.setattr("src.pipeline.run_walk_forward.WalkForwardValidator", lambda **kwargs: mock_validator)

    mock_logger = MagicMock()
    monkeypatch.setattr("src.pipeline.run_walk_forward.logger", mock_logger)

    run_walk_forward_validation("config.yaml")

    # Should call logger.info for each summary metric
    assert mock_logger.info.call_count == len(fake_results["summary"])
    mock_logger.info.assert_any_call("Walk-forward validation metric - rmse: 0.1000")
    mock_logger.info.assert_any_call("Walk-forward validation metric - mae: 0.0500")


def test_run_walk_forward_validation_empty_summary(monkeypatch):
    # Edge case: empty summary dict
    fake_config = {"some_param": 1}
    fake_results = {"summary": {}, "details": {}}

    monkeypatch.setattr("src.pipeline.run_walk_forward.load_config", lambda path: fake_config)
    mock_validator = MagicMock()
    mock_validator.run_validation.return_value = fake_results
    monkeypatch.setattr("src.pipeline.run_walk_forward.WalkForwardValidator", lambda **kwargs: mock_validator)

    results = run_walk_forward_validation("config.yaml")
    assert results == fake_results
    mock_validator.run_validation.assert_called_once()
