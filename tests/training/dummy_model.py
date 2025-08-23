import pickle
import numpy as np

class DummyModel:
    def __init__(self, params=None):
        self.params = params or {}
        self.is_trained = False

    def train(self, X, y, X_val=None, y_val=None):
        self.is_trained = True
        self.last_train_shape = (X.shape, y.shape)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"trained": self.is_trained}, f)
