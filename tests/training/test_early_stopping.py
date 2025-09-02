import os
import tempfile
import torch
import pytest
from src.training.optimizers.early_stopping import EarlyStopping


class DummyModel(torch.nn.Module):
    """
    Minimal PyTorch model used for testing EarlyStopping.
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)


@pytest.fixture
def dummy_model() -> DummyModel:
    """
    Fixture to provide a fresh DummyModel for each test.
    """
    return DummyModel()


@pytest.fixture
def temp_checkpoint_path(tmp_path) -> str:
    """
    Fixture that provides a temporary checkpoint path for saving models.
    """
    return tmp_path / "best_model.pt"


def test_improvement_resets_counter_and_saves_checkpoint(dummy_model, temp_checkpoint_path):
    """
    Test that when validation loss improves:
    - Counter is reset
    - Best loss is updated
    - Model checkpoint is saved
    """
    es = EarlyStopping(patience=3, checkpoint_path=str(temp_checkpoint_path))
    es(val_loss=0.5, model=dummy_model)

    # Best loss updated
    assert es.best_loss == pytest.approx(0.5)
    # Counter reset
    assert es.counter == 0
    # Checkpoint saved
    assert os.path.exists(temp_checkpoint_path)


def test_no_improvement_increments_counter(dummy_model, temp_checkpoint_path):
    """
    Test that when no improvement occurs, counter increments.
    """
    es = EarlyStopping(patience=2, checkpoint_path=str(temp_checkpoint_path))
    es.best_loss = 0.5

    es(val_loss=0.6, model=dummy_model)  # worse loss
    assert es.counter == 1
    assert not es.early_stop


def test_triggers_early_stopping_after_patience(dummy_model, temp_checkpoint_path):
    """
    Test that EarlyStopping triggers after exceeding patience.
    """
    es = EarlyStopping(patience=2, checkpoint_path=str(temp_checkpoint_path))
    es.best_loss = 0.5

    # No improvement for two consecutive epochs
    es(val_loss=0.6, model=dummy_model)
    es(val_loss=0.7, model=dummy_model)
    assert es.early_stop is True


def test_improvement_resets_counter_after_no_improvement(dummy_model, temp_checkpoint_path):
    """
    Test that an improvement after some non-improving epochs resets the counter.
    """
    es = EarlyStopping(patience=3, checkpoint_path=str(temp_checkpoint_path))
    es.best_loss = 0.5

    # No improvement first
    es(val_loss=0.55, model=dummy_model)
    assert es.counter == 1

    # Improvement next
    es(val_loss=0.4, model=dummy_model)
    assert es.counter == 0
    assert es.best_loss == pytest.approx(0.4)


def test_checkpoint_directory_is_created(dummy_model):
    """
    Test that checkpoint directory is created if it does not exist.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "nested/dir/best_model.pt")
        es = EarlyStopping(checkpoint_path=checkpoint_path)

        es(val_loss=0.3, model=dummy_model)
        assert os.path.exists(checkpoint_path)
