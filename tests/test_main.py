import sys
import pytest
from unittest.mock import patch, MagicMock
import builtins

import main


@pytest.fixture(autouse=True)
def reset_sys_argv():
    """Reset sys.argv after each test to avoid bleed-over."""
    old_argv = sys.argv.copy()
    yield
    sys.argv = old_argv


def test_validate_config_path_file_not_found(tmp_path):
    """Ensure FileNotFoundError is raised when config file does not exist."""
    non_existent = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        main.validate_config_path(str(non_existent))


@patch("main.start_api")
def test_main_serve_command(mock_start_api):
    """Test that 'serve' command triggers API startup."""
    sys.argv = ["main.py", "serve", "--host", "127.0.0.1", "--port", "9000"]

    main.main()

    mock_start_api.assert_called_once_with(host="127.0.0.1", port=9000)


@patch("main.PipelineOrchestrator")
def test_main_train_command(mock_orchestrator, tmp_path):
    """Test 'train' command dispatches to orchestrator with correct config."""
    config_file = tmp_path / "train_config.yaml"
    config_file.write_text("dummy: true")

    sys.argv = ["main.py", "train", "--config", str(config_file)]

    mock_instance = mock_orchestrator.return_value
    mock_instance.run_pipeline = MagicMock()

    main.main()

    mock_instance.run_pipeline.assert_called_once_with(["train"])


@patch("main.PipelineOrchestrator")
def test_main_pipeline_command(mock_orchestrator, tmp_path):
    """Test 'pipeline' command runs full orchestrated pipeline."""
    config_file = tmp_path / "full_config.yaml"
    config_file.write_text("dummy: true")

    sys.argv = ["main.py", "pipeline", "--config", str(config_file)]

    mock_instance = mock_orchestrator.return_value
    mock_instance.run_pipeline = MagicMock()

    main.main()

    # For full pipeline (no explicit steps)
    mock_instance.run_pipeline.assert_called_once_with()


def test_main_invalid_command(monkeypatch):
    """Test that argparse exits on invalid command."""
    sys.argv = ["main.py", "invalid"]

    with pytest.raises(SystemExit):
        main.main()
