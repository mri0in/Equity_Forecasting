
import pytest

class DummyModel:
    """A lightweight dummy model for testing ensemble modules."""
    def __init__(self):
        self.is_trained = False

    def fit(self, X, y):
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return [0] * len(X)

@pytest.fixture
def dummy_model():
    """Fixture that provides a fresh DummyModel instance."""
    return DummyModel()
