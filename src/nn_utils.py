"""
Shared neural network utilities for all models.

Contains common components used across different model architectures.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information.

    Shared by Transformer, TFA, and other sequence models.
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class BaseNeuralPredictor:
    """
    Base class for PyTorch-based predictors with common training logic.

    Reduces code duplication across Transformer, MLP, and TFA models.
    """

    def __init__(
        self,
        device: str = 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 50,
        early_stopping_patience: int = 5,
    ):
        self.device = torch.device(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        self.model = None  # To be set by subclass
        self.optimizer = None
        self.training_history = []
        self.best_model_state = None

    def _train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        verbose: bool = False,
    ) -> float:
        """
        Train for one epoch using mini-batches.

        Returns average training loss.
        """
        self.model.train()
        indices = np.random.permutation(len(X_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            self.optimizer.zero_grad()
            loss = self._compute_batch_loss(X_batch, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        return epoch_loss / n_batches

    def _compute_batch_loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch. To be implemented by subclass."""
        raise NotImplementedError

    def _validate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> float:
        """
        Compute validation loss.

        Returns validation loss.
        """
        self.model.eval()
        with torch.no_grad():
            val_loss = self._compute_batch_loss(X_val, y_val)
        return val_loss.item()

    def _early_stopping_check(
        self,
        val_loss: float,
        best_val_loss: float,
        patience_counter: int,
        verbose: bool = False,
    ) -> tuple[float, int, bool]:
        """
        Check early stopping condition.

        Returns:
            best_val_loss: Updated best validation loss
            patience_counter: Updated patience counter
            should_stop: Whether to stop training
        """
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            self.best_model_state = {
                k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
            }
        else:
            patience_counter += 1

        should_stop = patience_counter >= self.early_stopping_patience

        if should_stop and verbose:
            logger.info(f"Early stopping triggered")

        return best_val_loss, patience_counter, should_stop

    def fit_with_validation(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        verbose: bool = True,
    ):
        """
        Standard training loop with validation and early stopping.
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            train_loss = self._train_epoch(X_train, y_train, verbose)

            # Validation
            val_loss = self._validate(X_val, y_val)

            # Record history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
            })

            # Logging
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )

            # Early stopping
            best_val_loss, patience_counter, should_stop = self._early_stopping_check(
                val_loss, best_val_loss, patience_counter, verbose
            )

            if should_stop:
                self.model.load_state_dict(self.best_model_state)
                break

        # Load best model if early stopping didn't trigger
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
