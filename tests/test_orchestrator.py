import os
import pytest
import types

import src.pipeline.orchestrator as orchestrator


class DummyConfig:
    """Minimal fake pipeline config for testing orchestrator."""

    def __init__(self):
        self.tasks = ["train", "predict"]
        self.retries = 1
        self.strict = True


class DummyFullConfig:
    """Fake FullConfig with only pipeline section."""

    def __init__(self):
        self.pipeline = DummyConfig()


@pytest.fixture
def fake_orchestrator(monkeypatch, tmp_path):
    """Fixture that patches config loading and returns a PipelineOrchestrator."""
    # Patch load_typed_config to always return DummyFullConfig
    monkeypatch.setattr(orchestrator, "load_typed_config", lambda _: DummyFullConfig())

    # Patch task functions with fakes that record calls
    calls = {}

    def fake_stage(config_path):
        calls[config_path] = True

    monkeypatch.setattr(orchestrator, "run_training", fake_stage)
    monkeypatch.setattr(orchestrator, "run_prediction", fake_stage)
    monkeypatch.setattr(orchestrator, "run_optimizer", fake_stage)
    monkeypatch.setattr(orchestrator, "run_ensemble", fake_stage)
    monkeypatch.setattr(orchestrator, "run_walk_forward", fake_stage)

    orch = orchestrator.PipelineOrchestrator(config_path="dummy.yaml")
    orch._calls = calls  # attach for inspection
    return orch


def test_run_single_task_success(fake_orchestrator):
    """PipelineOrchestrator.run_task should call the correct wrapper when task is valid."""
    fake_orchestrator.run_task("train")
    assert "dummy.yaml" in fake_orchestrator._calls


def test_run_task_unknown(fake_orchestrator, caplog):
    """Unknown tasks should be skipped with a warning."""
    fake_orchestrator.run_task("unknown_task")
    assert "Unknown task" in caplog.text


def test_run_pipeline_custom_tasks(fake_orchestrator):
    """run_pipeline should execute only the specified tasks."""
    fake_orchestrator.run_pipeline(tasks=["predict"])
    assert "dummy.yaml" in fake_orchestrator._calls


def test_run_pipeline_from_config(fake_orchestrator):
    """run_pipeline should use config-defined tasks when none are passed."""
    fake_orchestrator.run_pipeline()
    # Both "train" and "predict" should run from DummyConfig
    assert "dummy.yaml" in fake_orchestrator._calls


def test_run_task_retries_and_strict(monkeypatch, fake_orchestrator):
    """If a task fails, it should retry and eventually raise if strict=True."""
    attempts = {"count": 0}

    def failing_stage(_):
        attempts["count"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator, "run_training", failing_stage)

    with pytest.raises(RuntimeError):
        fake_orchestrator.run_task("train", retries=2, strict=True)

    assert attempts["count"] == 2  # retried twice


def test_run_task_retries_non_strict(monkeypatch, fake_orchestrator, caplog):
    """If strict=False, it should log a warning instead of raising after retries."""
    attempts = {"count": 0}

    def failing_stage(_):
        attempts["count"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator, "run_training", failing_stage)

    fake_orchestrator.run_task("train", retries=2, strict=False)

    assert attempts["count"] == 2
    assert "Skipping failed task" in caplog.text


def test_task_marker_skips(monkeypatch, fake_orchestrator, tmp_path):
    """If a task marker file exists, the task should be skipped."""
    marker = orchestrator.TASK_MARKERS["train"]
    marker_path = tmp_path / ".train_complete"
    monkeypatch.setitem(orchestrator.TASK_MARKERS, "train", str(marker_path))

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("done")

    fake_orchestrator.run_task("train")
    # Nothing should be added to calls since it skipped
    assert fake_orchestrator._calls == {}
