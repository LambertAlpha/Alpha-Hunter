"""
Model architectures for return prediction.

Implements Transformer encoder and baseline models (Ridge, Random Forest, MLP).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Base Predictor Interface
# ============================================================================

class BasePredictor:
    """Base class for all predictors."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the model."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {}


# ============================================================================
# Transformer Model
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
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


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for sequence modeling.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension (number of PCA components)
    d_model : int, default=64
        Model dimension (embedding size)
    nhead : int, default=4
        Number of attention heads
    num_layers : int, default=2
        Number of transformer layers
    dim_feedforward : int, default=256
        Dimension of feedforward network
    dropout : float, default=0.1
        Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection: d_model -> 1 (return prediction)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape [batch_size, seq_len, input_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: shape [batch_size]
            (optional) attention_weights: list of attention matrices
        """
        # Project input to d_model
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        if return_attention:
            # Store attention weights (this is simplified; real implementation needs hooks)
            encoded = self.transformer_encoder(x)
            attention_weights = None  # Placeholder
        else:
            encoded = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        
        # Global average pooling over sequence
        pooled = encoded.mean(dim=1)  # [batch, d_model]
        
        # Output projection
        output = self.output_head(pooled).squeeze(-1)  # [batch]
        
        if return_attention:
            return output, attention_weights
        else:
            return output


class TransformerPredictor(BasePredictor):
    """
    Transformer-based return predictor with PyTorch.
    
    Parameters
    ----------
    input_dim : int
        Number of input features (PCA components)
    d_model : int, default=64
        Model embedding dimension
    nhead : int, default=4
        Number of attention heads
    num_layers : int, default=2
        Number of transformer layers
    dim_feedforward : int, default=256
        Feedforward dimension
    dropout : float, default=0.1
        Dropout rate
    lr : float, default=1e-3
        Learning rate
    weight_decay : float, default=1e-4
        L2 regularization
    batch_size : int, default=128
        Training batch size
    epochs : int, default=50
        Maximum training epochs
    early_stopping_patience : int, default=5
        Early stopping patience
    device : str, default='cpu'
        Device to use ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        device: str = 'cpu',
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device)
        
        # Initialize model
        self.model = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Scaler for input normalization
        self.scaler = StandardScaler()
        
        self.training_history = []
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Fit the Transformer model.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, seq_len, n_features)
            Training sequences
        y : np.ndarray, shape (n_samples,)
            Training targets
        X_val : np.ndarray, optional
            Validation sequences
        y_val : np.ndarray, optional
            Validation targets
        verbose : bool, default=True
            Whether to print training progress
        """
        # Normalize inputs
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_flat_scaled = self.scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, seq_len, n_features)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_scaled).to(self.device)
        y_train = torch.FloatTensor(y).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(-1, n_features)
            X_val_flat_scaled = self.scaler.transform(X_val_flat)
            X_val_scaled = X_val_flat_scaled.reshape(X_val.shape[0], seq_len, n_features)
            
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = nn.MSELoss()(predictions, y_batch)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = nn.MSELoss()(val_predictions, y_val_tensor).item()
                
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                })
                
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                })
                
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, seq_len, n_features)
            Input sequences
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted returns
        """
        self.model.eval()
        
        # Normalize
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_flat_scaled = self.scaler.transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, seq_len, n_features)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'model_type': 'Transformer',
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }


# ============================================================================
# Baseline Models
# ============================================================================

class RidgePredictor(BasePredictor):
    """
    Ridge regression baseline.
    
    Flattens sequence into features: [t-11, t-10, ..., t-1].
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fit Ridge regression.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, seq_len, n_features)
            Training sequences
        y : np.ndarray, shape (n_samples,)
            Training targets
        """
        # Flatten sequences
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        
        self.model.fit(X_scaled, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)
    
    def get_params(self) -> Dict[str, Any]:
        return {'model_type': 'Ridge', 'alpha': self.alpha}


class RandomForestPredictor(BasePredictor):
    """
    Random Forest baseline.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees
    max_depth : int, optional
        Maximum tree depth
    min_samples_split : int, default=10
        Minimum samples to split node
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, default=42
        Random seed
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 10,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Random Forest."""
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'model_type': 'RandomForest',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
        }


class MLPPredictor(BasePredictor):
    """
    Multi-layer Perceptron baseline with PyTorch.
    
    Parameters
    ----------
    input_dim : int
        Flattened input dimension (seq_len * n_features)
    hidden_dims : list of int, default=[256, 128, 64]
        Hidden layer dimensions
    dropout : float, default=0.2
        Dropout rate
    lr : float, default=1e-3
        Learning rate
    weight_decay : float, default=1e-4
        L2 regularization
    batch_size : int, default=128
        Training batch size
    epochs : int, default=50
        Maximum training epochs
    device : str, default='cpu'
        Device to use
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 50,
        device: str = 'cpu',
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)
        
        # Build network
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scaler = StandardScaler()
        self.training_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False, **kwargs):
        """Fit MLP."""
        # Flatten and normalize
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        
        X_train = torch.FloatTensor(X_scaled).to(self.device)
        y_train = torch.FloatTensor(y).to(self.device)
        
        for epoch in range(self.epochs):
            self.model.train()
            indices = np.random.permutation(len(X_train))
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                self.optimizer.zero_grad()
                predictions = self.model(X_batch).squeeze(-1)
                loss = nn.MSELoss()(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_loss = train_loss / n_batches
            self.training_history.append({'epoch': epoch + 1, 'train_loss': avg_loss})
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze(-1)
        
        return predictions.cpu().numpy()
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'model_type': 'MLP',
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }

