import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_meta_features(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load meta model features and targets from a CSV file.

    Args:
        path (str): Path to the CSV file containing features and targets.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X and target vector y.
    """
    logger.info(f"Loading meta features from: {path}")
    df = pd.read_csv(path)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X, y


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print and log evaluation metrics.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted values by the meta model.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    logger.info(f"Evaluation Metrics:")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot actual vs predicted values.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(y_true)), y=y_true, label="Actual")
    sns.lineplot(x=range(len(y_pred)), y=y_pred, label="Predicted")
    plt.title("Meta Model Predictions vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_model(model_path: str):
    """
    Load a trained meta model.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        Loaded model object.
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model
