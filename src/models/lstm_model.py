# src/models/lstm_model.py
import os
import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseModel
from src.optimizers.early_stopping import EarlyStopping


class LSTMModel(BaseModel):
    """
    LSTM model for equity forecasting using PyTorch.
    """

    class _LSTMNet(nn.Module):
        """
        Internal PyTorch LSTM network definition.
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout: float,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            x: (batch_size, seq_len, input_size)
            """
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]   # (batch_size, hidden_size)
            return self.fc(last_output)        # (batch_size, output_size)

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------
    def __init__(self, model_params: Optional[Dict[str, Any]] = None) -> None:
        model_params = model_params or {}
        super().__init__(model_params)

        self.input_size = self.model_params.get("input_size", 1)
        self.hidden_size = self.model_params.get("hidden_size", 64)
        self.num_layers = self.model_params.get("num_layers", 2)
        self.dropout = self.model_params.get("dropout", 0.2)
        self.batch_size = self.model_params.get("batch_size", 32)
        self.epochs = self.model_params.get("epochs", 20)
        self.learning_rate = self.model_params.get("learning_rate", 0.001)
        self.output_size = self.model_params.get("output_size", 1)
        self.lookback = self.model_params.get("lookback", 20)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("Using device: %s", self.device)

        self.model = self._LSTMNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout=self.dropout,
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.early_stopper: Optional[EarlyStopping] = None
        self.scheduler = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train the LSTM model.
        """

        train_dataset = TensorDataset( 
            torch.from_numpy(X_train).float(), 
            torch.from_numpy(y_train).float(), 
        )

        train_loader = DataLoader( train_dataset, batch_size=self.batch_size, shuffle=True, )

        val_loader = None

        if X_val is not None and y_val is not None:

            val_dataset = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float(),
            )
            val_loader = DataLoader( val_dataset, batch_size=self.batch_size, shuffle=False, )

            self.early_stopper = EarlyStopping(
                patience=5,
                delta=1e-4,
                verbose=True,
                checkpoint_path="best_lstm_model.pt",
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            )

        self.logger.info("Starting training for %d epochs...", self.epochs)

        # --------------------------------------------------------------
        # Epoch loop
        # --------------------------------------------------------------
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)  

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                loss = self.criterion(outputs, batch_y)

                if torch.isnan(loss):
                    raise RuntimeError("NaN loss detected during training")

                loss.backward()

                # Gradient clipping (LSTM safety)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            avg_train_loss = epoch_loss / len(train_loader.dataset)
            self.logger.info(
                "Epoch [%d/%d] - Train Loss: %.6f",
                epoch,
                self.epochs,
                avg_train_loss,
            )

            # ----------------------------------------------------------
            # Validation
            # ----------------------------------------------------------
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for val_X, val_y in val_loader:
                        val_X = val_X.to(self.device)
                        val_y = val_y.to(self.device).unsqueeze(1)  # FIX

                        val_outputs = self.model(val_X)
                        loss = self.criterion(val_outputs, val_y)

                        if torch.isnan(loss):
                            raise RuntimeError("NaN loss detected during validation")

                        val_loss += loss.item() * val_X.size(0)

                avg_val_loss = val_loss / len(val_loader.dataset)
                self.logger.info(
                    "Epoch [%d/%d] - Val Loss: %.6f",
                    epoch,
                    self.epochs,
                    avg_val_loss,
                )

                self.scheduler.step(avg_val_loss)
                self.early_stopper(avg_val_loss, self.model)

                if self.early_stopper.early_stop:
                    self.logger.info("Early stopping triggered.")
                    break

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        weights_path = os.path.join(directory, "lstm_weights.pt")
        torch.save(self.model.state_dict(), weights_path)

        params_path = os.path.join(directory, "model_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(self.model_params, f, indent=4)

        self.logger.info("Model saved to %s", directory)

    @classmethod
    def load_model(cls, directory: str) -> "LSTMModel":
        params_path = os.path.join(directory, "model_params.json")
        with open(params_path, "r", encoding="utf-8") as f:
            model_params = json.load(f)

        instance = cls(model_params)

        weights_path = os.path.join(directory, "lstm_weights.pt")
        state_dict = torch.load(weights_path, map_location=instance.device)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()

        return instance

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_test).float().to(self.device)
            outputs = self.model(inputs)
            return outputs.cpu().numpy()

    def get_params(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "device": str(self.device),
            "lookback": self.lookback,
        }
