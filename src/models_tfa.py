"""
Temporal Factor Autoencoder (TFA) - Core Innovation

Learns time-varying factor importance through attention-based reconstruction.

Key Features:
1. Dynamic Factor Weighting: Attention generates time-varying weights for PCA factors
2. Encoder-Decoder Architecture: Ensures information preservation via reconstruction
3. Temporal Smoothness: Regularizes weights for interpretable transitions
4. Multi-task Learning: Joint optimization of prediction and reconstruction

Reference: 
    Inspired by Gu, Kelly, Xiu (2020) but extended with temporal dynamics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .nn_utils import PositionalEncoding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalFactorAutoencoder(nn.Module):
    """
    Temporal Factor Autoencoder for learning dynamic factor importance.
    
    Architecture:
        PCA Factors → Encoder → Dynamic Weights → Latent Factors
                          ↓
                      Decoder (reconstruct PCA)
                          ↓
                      Predictor (return prediction)
    
    Parameters
    ----------
    n_pca_factors : int, default=11
        Number of PCA components
    seq_len : int, default=36
        Sequence length (months)
    d_model : int, default=128
        Model dimension
    n_heads : int, default=8
        Number of attention heads
    n_encoder_layers : int, default=4
        Number of encoder layers
    n_decoder_layers : int, default=2
        Number of decoder layers
    n_latent_factors : int, default=5
        Number of learned latent factors
    dropout : float, default=0.1
        Dropout rate
    n_classes : int, default=5
        Number of return quantiles for classification
    """
    
    def __init__(
        self,
        n_pca_factors: int = 11,
        seq_len: int = 36,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        n_latent_factors: int = 5,
        dropout: float = 0.1,
        n_classes: int = 5,
    ):
        super().__init__()
        
        self.n_pca_factors = n_pca_factors
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_latent_factors = n_latent_factors
        self.n_classes = n_classes
        
        # ===== Input Projection =====
        self.input_projection = nn.Linear(n_pca_factors, d_model)
        
        # ===== Positional Encoding =====
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        # ===== Transformer Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
            enable_nested_tensor=False,  # Disable nested tensor when norm_first=True
        )
        
        # ===== Dynamic Factor Weight Generator (KEY INNOVATION!) =====
        self.factor_weight_generator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_pca_factors),
            nn.Softmax(dim=-1)  # Generates probability distribution over factors
        )
        
        # ===== Latent Factor Extractor =====
        self.latent_projector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_latent_factors)
        )
        
        # ===== Transformer Decoder =====
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_decoder_layers,
        )
        
        # ===== Reconstruction Head =====
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_pca_factors)
        )
        
        # ===== Prediction Head =====
        self.predictor = nn.Sequential(
            nn.Linear(n_latent_factors, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
        self._init_weights()
        
        logger.info(f"Initialized TFA with {self.count_parameters():,} parameters")
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self, 
        pca_seq: torch.Tensor,
        return_attention: bool = False,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through TFA.
        
        Args:
            pca_seq: (batch, seq_len, n_pca_factors) - PCA factor sequences
            return_attention: Whether to return attention weights
            return_all: Whether to return all intermediate outputs
        
        Returns:
            pred_logits: (batch, n_classes) - Predicted return quantiles
            reconstructed: (batch, seq_len, n_pca_factors) - Reconstructed PCA
            factor_weights: (batch, seq_len, n_pca_factors) - Dynamic weights (if return_all)
            latent_factors: (batch, n_latent_factors) - Learned factors (if return_all)
        """
        batch_size, seq_len, n_features = pca_seq.shape
        
        # 1. Project input to d_model
        x = self.input_projection(pca_seq)  # (batch, seq_len, d_model)
        
        # 2. Add positional encoding
        x = self.pos_encoder(x)
        
        # 3. Encode: Learn temporal patterns
        encoded = self.encoder(x)  # (batch, seq_len, d_model)
        
        # 4. Generate dynamic factor weights (KEY INNOVATION!)
        factor_weights = self.factor_weight_generator(encoded)
        # (batch, seq_len, n_pca_factors)
        # Each timestep has a distribution over PCA factors
        
        # 5. Apply dynamic weighting to original PCA factors
        weighted_pca = pca_seq * factor_weights
        # (batch, seq_len, n_pca_factors)
        
        # 6. Extract latent factors from last timestep
        last_encoded = encoded[:, -1, :]  # (batch, d_model)
        latent_factors = self.latent_projector(last_encoded)
        # (batch, n_latent_factors)
        
        # 7. Decode: Reconstruct original PCA factors
        decoded = self.decoder(
            tgt=x,           # Target sequence
            memory=encoded   # Encoder output (memory)
        )  # (batch, seq_len, d_model)
        
        reconstructed = self.recon_head(decoded)
        # (batch, seq_len, n_pca_factors)
        
        # 8. Predict return quantile
        pred_logits = self.predictor(latent_factors)
        # (batch, n_classes)
        
        if return_all:
            return pred_logits, reconstructed, factor_weights, latent_factors
        elif return_attention:
            return pred_logits, reconstructed, factor_weights
        else:
            return pred_logits, reconstructed
    
    def compute_loss(
        self,
        pca_seq: torch.Tensor,
        y_true: torch.Tensor,
        alpha: float = 0.1,
        beta: float = 0.05,
        gamma: float = 0.01
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss with regularization.
        
        Args:
            pca_seq: (batch, seq_len, n_pca_factors)
            y_true: (batch,) - Return quantile labels [0, n_classes)
            alpha: Weight for reconstruction loss
            beta: Weight for temporal smoothness loss
            gamma: Weight for orthogonality loss
        
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary of individual loss components
        """
        # Forward pass
        pred_logits, recon, weights, latent = self.forward(
            pca_seq, return_all=True
        )
        
        # Loss 1: Classification loss (PRIMARY)
        pred_loss = F.cross_entropy(pred_logits, y_true)
        
        # Loss 2: Reconstruction loss (AUXILIARY)
        # Ensures encoder preserves information
        recon_loss = F.mse_loss(recon, pca_seq)
        
        # Loss 3: Temporal smoothness loss (REGULARIZATION)
        # Prevents erratic weight changes, enhances interpretability
        if self.seq_len > 1:
            weight_diff = weights[:, 1:, :] - weights[:, :-1, :]
            smooth_loss = (weight_diff ** 2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=pca_seq.device)
        
        # Loss 4: Orthogonality loss (OPTIONAL)
        # Encourages learned latent factors to be independent
        if self.n_latent_factors > 1:
            # Compute covariance matrix
            latent_centered = latent - latent.mean(dim=0, keepdim=True)
            latent_cov = torch.matmul(latent_centered.T, latent_centered)
            latent_cov = latent_cov / (latent.size(0) - 1)
            
            # Orthogonality: covariance should be diagonal
            eye = torch.eye(
                self.n_latent_factors, 
                device=latent.device
            )
            # Normalize by variance to make diagonal elements ~1
            latent_std = torch.sqrt(torch.diag(latent_cov))
            latent_corr = latent_cov / (latent_std.unsqueeze(1) * latent_std.unsqueeze(0) + 1e-8)
            
            ortho_loss = ((latent_corr - eye) ** 2).mean()
        else:
            ortho_loss = torch.tensor(0.0, device=pca_seq.device)
        
        # Total loss
        total_loss = (
            pred_loss + 
            alpha * recon_loss + 
            beta * smooth_loss +
            gamma * ortho_loss
        )
        
        # Return loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'prediction': pred_loss.item(),
            'reconstruction': recon_loss.item(),
            'smoothness': smooth_loss.item(),
            'orthogonality': ortho_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def predict_proba(self, pca_seq: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            pca_seq: (batch, seq_len, n_pca_factors)
        
        Returns:
            probs: (batch, n_classes) - Probability distribution
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(pca_seq)
            probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_factor_weights(self, pca_seq: torch.Tensor) -> torch.Tensor:
        """
        Extract dynamic factor weights for analysis.
        
        Args:
            pca_seq: (batch, seq_len, n_pca_factors)
        
        Returns:
            weights: (batch, seq_len, n_pca_factors)
        """
        self.eval()
        with torch.no_grad():
            _, _, weights = self.forward(pca_seq, return_attention=True)
        return weights
    
    def get_latent_factors(self, pca_seq: torch.Tensor) -> torch.Tensor:
        """
        Extract learned latent factors.
        
        Args:
            pca_seq: (batch, seq_len, n_pca_factors)
        
        Returns:
            latent: (batch, n_latent_factors)
        """
        self.eval()
        with torch.no_grad():
            _, _, _, latent = self.forward(pca_seq, return_all=True)
        return latent


class TFAPredictor:
    """
    Wrapper class for TFA that handles training and prediction.
    Compatible with the existing training framework.
    """
    
    def __init__(
        self,
        n_pca_factors: int = 11,
        seq_len: int = 36,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        n_latent_factors: int = 5,
        dropout: float = 0.1,
        n_classes: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        alpha: float = 0.1,  # Reconstruction weight
        beta: float = 0.05,  # Smoothness weight
        gamma: float = 0.01,  # Orthogonality weight
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        
        # Model hyperparameters
        self.n_pca_factors = n_pca_factors
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize model
        self.model = TemporalFactorAutoencoder(
            n_pca_factors=n_pca_factors,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_latent_factors=n_latent_factors,
            dropout=dropout,
            n_classes=n_classes,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        self.training_history = []
        self.best_model_state = None
        self.quantile_boundaries = None  # 存储全局quantile边界
        self.scaler = StandardScaler()  # 数据标准化
    
    def _compute_quantile_boundaries(self, y: np.ndarray) -> np.ndarray:
        """
        Compute global quantile boundaries from training data.
        
        Args:
            y: (n_samples,) - Continuous returns
        
        Returns:
            boundaries: (n_classes + 1,) - Quantile boundaries
        """
        quantiles = np.linspace(0, 1, self.n_classes + 1)
        boundaries = np.quantile(y, quantiles)
        return boundaries
    
    def _prepare_labels(self, y: np.ndarray, boundaries: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Convert continuous returns to quantile labels using pre-computed boundaries.
        
        Args:
            y: (n_samples,) - Continuous returns
            boundaries: (n_classes + 1,) - Quantile boundaries (if None, use stored)
        
        Returns:
            labels: (n_samples,) - Quantile labels [0, n_classes)
        """
        if boundaries is None:
            boundaries = self.quantile_boundaries
        
        if boundaries is None:
            # Fallback: compute from current batch (not ideal but backward compatible)
            quantiles = np.linspace(0, 1, self.n_classes + 1)
            boundaries = np.quantile(y, quantiles)
            logger.warning("Using batch-level quantiles. Should compute global quantiles in fit().")
        
        # Assign labels based on boundaries
        labels = np.zeros_like(y, dtype=np.int64)
        for i in range(len(boundaries) - 1):
            if i == 0:
                # First bin: <= boundary[1]
                mask = y <= boundaries[i+1]
            elif i == len(boundaries) - 2:
                # Last bin: > boundary[-2]
                mask = y > boundaries[i]
            else:
                # Middle bins: boundaries[i] < y <= boundaries[i+1]
                mask = (y > boundaries[i]) & (y <= boundaries[i+1])
            labels[mask] = i
        
        return torch.LongTensor(labels)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Train the TFA model.
        
        Args:
            X: (n_samples, seq_len, n_features) - Training sequences
            y: (n_samples,) - Training targets (continuous returns)
            X_val: Validation sequences
            y_val: Validation targets
            verbose: Whether to print progress
        """
        # Compute global quantile boundaries from training data
        if self.quantile_boundaries is None:
            self.quantile_boundaries = self._compute_quantile_boundaries(y)
            if verbose:
                logger.info(f"Computed quantile boundaries: {self.quantile_boundaries}")
                logger.info(f"  Range: [{self.quantile_boundaries[0]:.4f}, {self.quantile_boundaries[-1]:.4f}]")
        
        # Normalize input data
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_flat_scaled = self.scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, seq_len, n_features)
        
        # Convert to tensors using global boundaries
        X_train = torch.FloatTensor(X_scaled).to(self.device)
        y_train = self._prepare_labels(y, boundaries=self.quantile_boundaries).to(self.device)
        
        if X_val is not None and y_val is not None:
            # Normalize validation data using training scaler
            n_val_samples, val_seq_len, val_n_features = X_val.shape
            X_val_flat = X_val.reshape(-1, val_n_features)
            X_val_flat_scaled = self.scaler.transform(X_val_flat)
            X_val_scaled = X_val_flat_scaled.reshape(n_val_samples, val_seq_len, val_n_features)
            
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            # Use same boundaries for validation
            y_val_tensor = self._prepare_labels(y_val, boundaries=self.quantile_boundaries).to(self.device)
            use_validation = True
        else:
            use_validation = False
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(
            range(self.epochs),
            desc="Epochs",
            disable=not verbose,
            ncols=100,
            leave=True
        )
        
        for epoch in epoch_pbar:
            self.model.train()
            
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            epoch_losses = {
                'total': [], 'prediction': [], 
                'reconstruction': [], 'smoothness': [], 'orthogonality': []
            }
            
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                # Forward and compute loss
                self.optimizer.zero_grad()
                loss, loss_dict = self.model.compute_loss(
                    X_batch, y_batch,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma
                )
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Log
                for key, value in loss_dict.items():
                    epoch_losses[key].append(value)
            
            # Average losses
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            
            # Validation
            if use_validation:
                self.model.eval()
                with torch.no_grad():
                    val_loss, val_loss_dict = self.model.compute_loss(
                        X_val_tensor, y_val_tensor,
                        alpha=self.alpha, beta=self.beta, gamma=self.gamma
                    )
                    val_loss_value = val_loss.item()
                
                # Learning rate scheduling
                self.scheduler.step(val_loss_value)
                
                # Record history
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_losses['total'],
                    'val_loss': val_loss_value,
                    **{f'train_{k}': v for k, v in avg_losses.items()},
                    **{f'val_{k}': v for k, v in val_loss_dict.items()},
                })
                
                # Update progress bar
                epoch_pbar.set_postfix({
                    'Train': f"{avg_losses['total']:.4f}",
                    'Val': f"{val_loss_value:.4f}",
                    'Best': f"{best_val_loss:.4f}",
                    'Patience': patience_counter
                })
                
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} | "
                        f"Train Loss: {avg_losses['total']:.4f} | "
                        f"Val Loss: {val_loss_value:.4f} | "
                        f"Pred: {avg_losses['prediction']:.4f} | "
                        f"Recon: {avg_losses['reconstruction']:.4f}"
                    )
                
                # Early stopping
                if val_loss_value < best_val_loss:
                    best_val_loss = val_loss_value
                    patience_counter = 0
                    self.best_model_state = {
                        k: v.cpu().clone() 
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    epoch_pbar.close()
                    break
            else:
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_losses['total'],
                    **{f'train_{k}': v for k, v in avg_losses.items()},
                })
                
                # Update progress bar
                epoch_pbar.set_postfix({
                    'Train': f"{avg_losses['total']:.4f}"
                })
                
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} | "
                        f"Loss: {avg_losses['total']:.4f}"
                    )
        
        # Close progress bar
        epoch_pbar.close()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions (return expected return, not class).
        
        Args:
            X: (n_samples, seq_len, n_features)
        
        Returns:
            predictions: (n_samples,) - Expected returns
        """
        self.model.eval()
        
        # Normalize input data using training scaler
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_flat_scaled = self.scaler.transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, seq_len, n_features)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            probs = self.model.predict_proba(X_tensor)
        
        # Convert class probabilities to expected return
        # Use actual quantile boundary midpoints as class values
        if self.quantile_boundaries is not None:
            # Compute midpoints of each quantile bin
            class_values = []
            for i in range(len(self.quantile_boundaries) - 1):
                midpoint = (self.quantile_boundaries[i] + self.quantile_boundaries[i+1]) / 2
                class_values.append(midpoint)
            class_values = torch.tensor(class_values, dtype=torch.float32, device=self.device)
        else:
            # Fallback: use wider range based on typical return distribution
            # Most stock returns are in [-0.3, 0.3] range monthly
            logger.warning("No quantile boundaries found. Using default range [-0.2, 0.2]")
            class_values = torch.linspace(-0.2, 0.2, self.n_classes, device=self.device)
        
        expected_returns = (probs * class_values).sum(dim=-1)
        return expected_returns.cpu().numpy()
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'model_type': 'TemporalFactorAutoencoder',
            'n_pca_factors': self.n_pca_factors,
            'seq_len': self.seq_len,
            'n_parameters': self.model.count_parameters(),
            'lr': self.lr,
            'batch_size': self.batch_size,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
        }

    def save(self, path: str):
        """Save best model weights if available, otherwise current weights."""
        state = self.best_model_state if self.best_model_state is not None else self.model.state_dict()
        torch.save(state, path)
