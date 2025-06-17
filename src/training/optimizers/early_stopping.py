import os
import numpy as np
import torch
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EarlyStopping:
    """
    Implements early stopping mechanism to halt training when validation loss stops improving.

    This is useful to prevent overfitting and unnecessary training epochs.

    Attributes:
        patience (int): Number of epochs to wait before stopping after no improvement.
        delta (float): Minimum change in validation loss to qualify as improvement.
        verbose (bool): If True, logs validation improvements.
        counter (int): Counts epochs with no improvement.
        best_loss (float): Best recorded validation loss.
        early_stop (bool): Whether early stopping should trigger.
        checkpoint_path (str): Where to save the best model.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 1e-4,
        verbose: bool = True,
        checkpoint_path: Optional[str] = "checkpoints/best_model.pt"
    ) -> None:
        """
        Initialize the early stopping monitor.

        Args:
            patience (int): Epochs to wait for improvement before stopping.
            delta (float): Minimum change to qualify as improvement.
            verbose (bool): Log info on improvements.
            checkpoint_path (str): Path to save the best model weights.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path

        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        """
        Check whether validation loss improved; if not, increment counter.

        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): The PyTorch model to checkpoint if improved.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self._save_checkpoint(model)
            self.counter = 0
            if self.verbose:
                logger.info(f"Validation loss improved to {val_loss:.6f}. Model checkpoint saved.")
        else:
            self.counter += 1
            logger.info(
                f"No improvement in validation loss. Counter: {self.counter}/{self.patience}."
            )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} consecutive non-improving epochs."
                )

    def _save_checkpoint(self, model: torch.nn.Module) -> None:
        """
        Save the current model state to the checkpoint path.

        Ensures the target directory exists before saving.

        Args:
            model (torch.nn.Module): Model to save.
        """
        dir_path = os.path.dirname(self.checkpoint_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        torch.save(model.state_dict(), self.checkpoint_path)
        logger.debug(f"Model checkpoint saved at: {self.checkpoint_path}")

