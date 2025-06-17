import logging
from typing import Any, Dict, Optional
import torch
import os
import json

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.models.base_model import BaseModel
from src.training.optimizers.early_stopping import EarlyStopping

class LSTMModel(BaseModel):
    """
    LSTM model for equity forecasting using PyTorch.

    Inherits from BaseModel and implements training and prediction functionality
    for time series forecasting tasks with LSTM architecture.
    """

    class _LSTMNet(nn.Module):
        """
        Internal PyTorch LSTM network definition.

        Args:
            input_size (int): Number of input features per timestep.
            hidden_size (int): Number of hidden units in LSTM layers.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Number of output features.
            dropout (float): Dropout probability between LSTM layers.
        """
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the LSTM network.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

            Returns:
                torch.Tensor: Output predictions of shape (batch_size, output_size).
            """
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]  # Output at last time step
            out = self.fc(last_output)        # map to output dimension
            return out

    def __init__(self, model_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize LSTMModel with hyperparameters.

        Args:
            model_params (Optional[Dict[str, Any]]): Dictionary of model hyperparameters.
				Expected keys:
                    - input_size (int): Number of input features per timestep.
                    - hidden_size (int): Number of hidden LSTM units.
                    - num_layers (int): Number of LSTM layers.
                    - output_size (int): Number of output features.
                    - dropout (float): Dropout rate between LSTM layers.
                    - batch_size (int): Batch size for training.
                    - epochs (int): Number of training epochs.
                    - learning_rate (float): Learning rate for optimizer.

        """
        super().__init__(model_params)

        self.input_size = self.model_params.get('input_size', 1)
        self.hidden_size = self.model_params.get('hidden_size', 50)
        self.num_layers = self.model_params.get('num_layers', 2)
        self.output_size = self.model_params.get('output_size', 1)
        self.dropout = self.model_params.get('dropout', 0.2)
        self.batch_size = self.model_params.get('batch_size', 32)
        self.epochs = self.model_params.get('epochs', 20)
        self.learning_rate = self.model_params.get('learning_rate', 0.001)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        self.model = self._LSTMNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout=self.dropout
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.early_stopper = None
        self.scheduler = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Train the LSTM model with optional validation, early stopping and LR scheduler.

        Args:
            X_train (np.ndarray): Training features, shape (num_samples, seq_len, input_size).
            y_train (np.ndarray): Training targets, shape (num_samples, output_size).
            X_val (Optional[np.ndarray]): Validation features (optional).
            y_val (Optional[np.ndarray]): Validation targets (optional).
        """
        self.model.train()

        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            self.early_stopper = EarlyStopping(
                patience=5,
                delta=1e-4,
                verbose=True,
                checkpoint_path="best_lstm_model.pt"
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2)
        else:
            val_loader = None

        self.logger.info(f"Starting training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            avg_train_loss = epoch_loss / len(train_loader.dataset)
            self.logger.info(f"Epoch [{epoch}/{self.epochs}] - Train Loss: {avg_train_loss:.6f}")

            # === Validation ===
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_X, val_y in val_loader:
                        val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                        val_outputs = self.model(val_X)
                        loss = self.criterion(val_outputs, val_y)
                        val_loss += loss.item() * val_X.size(0)

                avg_val_loss = val_loss / len(val_loader.dataset)
                self.logger.info(f"Epoch [{epoch}/{self.epochs}] - Val Loss: {avg_val_loss:.6f}")

                # Step scheduler and early stopping
                self.scheduler.step(avg_val_loss)
                self.early_stopper(avg_val_loss, self.model)

                if self.early_stopper.early_stop:
                    self.logger.info("Early stopping triggered.")
                    break


    def save_model(self, directory: str) -> None:
        """
        Save model weights and hyperparameters to the specified directory.

        Args:
            directory (str): Path to the directory where model will be saved.
        """
        os.makedirs(directory, exist_ok=True)

		# Save model weights
        weights_path = os.path.join(directory, "lstm_weights.pt")
        torch.save(self.model.state_dict(), weights_path)
        self.logger.info("Model weights saved to %s", weights_path)
		
		# Save model parameters
        params_path = os.path.join(directory, "model_params.json")
        with open(params_path, "w") as f:
            json.dump(self.model_params, f, indent=4)
        self.logger.info("Model hyperparameters saved to %s", params_path)

    @classmethod
    def load_model(cls, directory: str) -> "LSTMModel":
        """
        Load model weights and hyperparameters from a directory and return an instance.

        Args:
            directory (str): Path to the directory containing model files.

        Returns:
            LSTMModel: Restored model ready for inference.
        """
		# Load hyperparameters
        params_path = os.path.join(directory, "model_params.json")
        with open(params_path, "r") as f:
            model_params = json.load(f)
		# Initialize model instance
        instance = cls(model_params)
		
		# Load model weights
        weights_path = os.path.join(directory, "lstm_weights.pt")
        state_dict = torch.load(weights_path, map_location=instance.device)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()

        instance.logger.info("Model loaded from %s", directory)
        return instance

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions with the trained model.

        Args:
            X_test (np.ndarray): Test features, shape (num_samples, seq_len, input_size).

        Returns:
            np.ndarray: Predicted values, shape (num_samples, output_size).
        """
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_test).float().to(self.device)
            outputs = self.model(inputs)
            predictions = outputs.cpu().numpy()
        return predictions
    
def get_params(self) -> Dict[str, Any]:
    """
    Get a copy of the model's current hyperparameters.

    Returns:
        Dict[str, Any]: Dictionary containing model hyperparameters.
    """
    self.logger.debug("Fetching model parameters.")
    return {
        "input_size": self.input_size,
        "hidden_size": self.hidden_size,
        "num_layers": self.num_layers,
        "output_size": self.output_size,
        "dropout": self.dropout,
        "batch_size": self.batch_size,
        "epochs": self.epochs,
        "learning_rate": self.learning_rate,
        "device": str(self.device)
    }
