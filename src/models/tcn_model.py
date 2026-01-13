# src/models/tcn_model.py
"""
Temporal Convolutional Network (TCN) Model

Risk Axis Covered
-----------------
- Temporal dependency risk
- Lagged effect and delayed reaction risk
- Long-memory sequence patterns

What This Model Hedges Against
------------------------------
- Missed long-range temporal structure
- Inadequate lag modeling by tabular methods
- Sequence effects where order matters more than instantaneous value

Failure Modes / What It Does NOT Capture
----------------------------------------
- Cross-sectional interactions unless encoded
- Sparse signal environments with weak temporal structure
- Structural regime breaks without retraining

Role in Ensemble
----------------
Acts as a temporal specialist.
Complements tabular models by capturing sequence-driven alpha.
Often excels during trending or momentum-dominated regimes.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseModel


# ======================================================================
# Low-level TCN module (tabular-safe)
# ======================================================================
class TCN(nn.Module):
    """
    Minimal Conv1D network for tabular regression.

    Input shape:
        (batch_size, 1, n_features)

    Output shape:
        (batch_size,)
    """

    def __init__( self, *, hidden_dim: int = 32, ) -> None:

        
        super().__init__()
        # Kernel size = 1 → no sequence expansion or shrinkage
        # Padding = 0 → prevents broadcast mismatch
        self.net = nn.Sequential(
            nn.Conv1d( in_channels=1, out_channels=hidden_dim, kernel_size=1, padding=0, ),
            nn.ReLU(),
            nn.Conv1d( in_channels=hidden_dim, out_channels=1, kernel_size=1, padding=0, ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F)
        out = self.net(x)          # (Batch, 1, n_Features)
        out = out.mean(dim=2)      # aggregate over feature-sequence
        return out                 # (Batch,)


# ======================================================================
# Model wrapper (Pipeline-E compatible)
# ======================================================================
class TCNModel(BaseModel):
    """
    TCN wrapper compatible with Pipeline E.

    Accepts NumPy arrays from pooled dataset.
    """

    def __init__(self, model_params: Optional[dict] = None) -> None:
        model_params = model_params or {}
        
        super().__init__(model_params)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # -----------------------------
        # Hyperparameters
        # -----------------------------
        self.epochs: int = model_params.get("epochs", 20)
        self.lr: float = model_params.get("lr", 1e-3)
        self.batch_size: int = model_params.get("batch_size", 32)
        self.hidden_dim: int = model_params.get("hidden_dim", 32)

        self.model: Optional[TCN] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train TCN on pooled tabular features.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_features)
        y : np.ndarray
            Shape (n_samples,)
        """

        # -----------------------------
        # Tensor conversion
        # -----------------------------
        X_tensor = torch.tensor( X, dtype=torch.float32 ).unsqueeze(1)  # (B, 1, F)
        y_tensor = torch.tensor( y, dtype=torch.float32 )

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # -----------------------------
        # Model init
        # -----------------------------
        self.model = TCN( hidden_dim=self.hidden_dim ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

        loss_fn = nn.MSELoss()

        self.logger.info(
            "Training TCN | samples=%d | features=%d",
            X.shape[0],
            X.shape[1],
        )

        # -----------------------------
        # Training loop
        # -----------------------------
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                preds = self.model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            self.logger.info(
                "Epoch %d/%d | loss=%.6f",
                epoch + 1,
                self.epochs,
                epoch_loss / len(loader),
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_features)
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        self.model.eval()

        X_tensor = torch.tensor(
            X, dtype=torch.float32
        ).unsqueeze(1).to(self.device)

        with torch.no_grad():
            preds = self.model(X_tensor)

        return preds.cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")

        torch.save(self.model.state_dict(), path)
        self.logger.info("TCN model saved → %s", path)

    def load_model(self, path: str) -> None:
        self.model = TCN(
            hidden_dim=self.hidden_dim
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )

        self.logger.info("TCN model loaded ← %s", path)