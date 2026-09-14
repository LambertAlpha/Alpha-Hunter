"""
Temporal Factor Autoencoder (TFA) - experimental sequence rank model

Learns time-varying factor importance through attention-based reconstruction.

Key Features:
1. Dynamic Factor Weighting: Attention generates time-varying weights for PCA factors
2. Encoder-Decoder Architecture: Ensures information preservation via reconstruction
3. Temporal Smoothness: Regularizes weights for interpretable transitions
4. Multi-task Learning: Joint optimization of prediction and reconstruction

Reference: 
    Conceptually related to representation learning; not a replication of Gu/Kelly/Xiu.
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        factor_gating: bool = True,
    ):
        super().__init__()
        
        if min(n_pca_factors, seq_len, n_heads, n_encoder_layers, n_decoder_layers, n_latent_factors) < 1:
            raise ValueError("Architecture dimensions must be positive")
        if d_model < 2 or d_model % n_heads or n_classes < 2:
            raise ValueError("d_model must be divisible by n_heads; n_classes >= 2")
        self.factor_gating = factor_gating
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
        
        # ===== Dynamic Factor Weight Generator =====
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
        if pca_seq.ndim != 3 or pca_seq.shape[1:] != (self.seq_len, self.n_pca_factors):
            raise ValueError("Expected (batch, configured sequence length, PCA features)")
        
        # 1. Project input to d_model
        x = self.input_projection(pca_seq)  # (batch, seq_len, d_model)
        
        # 2. Add positional encoding
        x = self.pos_encoder(x)
        
        # 3. Encode: Learn temporal patterns
        encoded = self.encoder(x)  # (batch, seq_len, d_model)
        
        # 4. Generate dynamic factor weights
        factor_weights = self.factor_weight_generator(encoded)
        # (batch, seq_len, n_pca_factors)
        # Each timestep has a distribution over PCA factors
        
        # Residual feature gate. Uniform weights recover the ungated graph exactly.
        # This gives the weight head a path to prediction and reconstruction losses.
        context = encoded
        if self.factor_gating:
            deviation = (self.n_pca_factors * factor_weights - 1) * pca_seq
            context = encoded + F.linear(deviation, self.input_projection.weight)
        last_encoded = context[:, -1, :]
        latent_factors = self.latent_projector(last_encoded)
        # (batch, n_latent_factors)
        
        # 7. Decode: Reconstruct original PCA factors
        decoded = self.decoder(
            tgt=x,           # Target sequence
            memory=context   # Gated encoder memory
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
        if pca_seq.size(1) > 1:
            weight_diff = weights[:, 1:, :] - weights[:, :-1, :]
            smooth_loss = (weight_diff ** 2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=pca_seq.device)
        
        # Loss 4: Orthogonality loss (OPTIONAL)
        # Encourages learned latent factors to be independent
        if self.n_latent_factors > 1 and latent.size(0) > 1:
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
            latent_std = torch.sqrt(torch.diag(latent_cov).clamp_min(1e-8))
            latent_corr = latent_cov / (latent_std.unsqueeze(1) * latent_std.unsqueeze(0) + 1e-8)
            
            off_diagonal = latent_corr * (1 - eye)
            ortho_loss = off_diagonal.square().sum() / (self.n_latent_factors * (self.n_latent_factors - 1))
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
    """Fit a sequence classifier and return scores in training-label units.

    In this project labels are cross-sectional percentile ranks, so predictions
    are ranking scores, not calibrated economic returns. Each fit starts from
    this instance's initial weights and resets the optimizer/scaler/history.
    """
    def __init__(self, n_pca_factors: int = 11, seq_len: int = 36,
                 d_model: int = 128, n_heads: int = 8, n_encoder_layers: int = 4,
                 n_decoder_layers: int = 2, n_latent_factors: int = 5,
                 dropout: float = .1, n_classes: int = 5, lr: float = 1e-3,
                 weight_decay: float = 1e-4, batch_size: int = 128, epochs: int = 50,
                 early_stopping_patience: int = 5, alpha: float = .1,
                 beta: float = .05, gamma: float = .01, device: str = 'cpu',
                 factor_gating: bool = True):
        if min(batch_size, epochs, early_stopping_patience) < 1:
            raise ValueError("Batch size, epochs and patience must be positive")
        if not np.isfinite([lr, weight_decay, alpha, beta, gamma]).all() or lr <= 0 or min(weight_decay, alpha, beta, gamma) < 0:
            raise ValueError("Invalid learning rate or loss/decay weights")
        self.config: dict[str, Any] = dict(n_pca_factors=n_pca_factors, seq_len=seq_len, d_model=d_model,
                           n_heads=n_heads, n_encoder_layers=n_encoder_layers,
                           n_decoder_layers=n_decoder_layers, n_latent_factors=n_latent_factors,
                           dropout=dropout, n_classes=n_classes, lr=lr, weight_decay=weight_decay,
                           batch_size=batch_size, epochs=epochs, early_stopping_patience=early_stopping_patience,
                           alpha=alpha, beta=beta, gamma=gamma, device=device, factor_gating=factor_gating)
        self.n_pca_factors, self.seq_len, self.n_classes = n_pca_factors, seq_len, n_classes
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.epochs = batch_size, epochs
        self.early_stopping_patience = early_stopping_patience
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.device = torch.device(device)
        architecture = {key: self.config[key] for key in (
            'n_pca_factors', 'seq_len', 'd_model', 'n_heads', 'n_encoder_layers',
            'n_decoder_layers', 'n_latent_factors', 'dropout', 'n_classes', 'factor_gating')}
        self.model = TemporalFactorAutoencoder(**architecture).to(self.device)
        self._initial_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.training_history = []
        self.best_model_state = None
        self.best_epoch = None
        self.quantile_boundaries = None
        self.feature_names: Optional[list[str]] = None
        self.scaler = StandardScaler()

    def _validate_X(self, X: np.ndarray):
        if X.ndim != 3 or X.shape[1:] != (self.seq_len, self.n_pca_factors) or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("Expected nonempty finite (samples, seq_len, features) input")

    def _prepare_labels(self, y: np.ndarray, boundaries: Optional[np.ndarray] = None) -> torch.Tensor:
        boundaries = self.quantile_boundaries if boundaries is None else boundaries
        if boundaries is None:
            raise ValueError("Fit training quantile boundaries first")
        return torch.as_tensor(np.searchsorted(boundaries[1:-1], y, side='left'), dtype=torch.long)

    def transform_inputs(self, X: np.ndarray) -> np.ndarray:
        """Apply the exact fitted normalization used for training and interpretation."""
        self._validate_X(X)
        return np.asarray(self.scaler.transform(X.reshape(-1, self.n_pca_factors))).reshape(X.shape).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, verbose: bool = True):
        self._validate_X(X)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("Expected one finite target per training sequence")
        if (X_val is None) != (y_val is None):
            raise ValueError("Validation inputs and labels must be supplied together")
        if X_val is not None:
            assert y_val is not None
            self._validate_X(X_val)
            if y_val.shape != (len(X_val),) or not np.isfinite(y_val).all():
                raise ValueError("Invalid validation targets")
        self.model.load_state_dict(self._initial_state)
        self.training_history = []
        self.best_model_state = None
        self.best_epoch = None
        self.quantile_boundaries = np.quantile(y, np.linspace(0, 1, self.n_classes + 1))
        self.scaler.fit(X.reshape(-1, self.n_pca_factors))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=.5)
        # Keep the full panel on CPU; transfer individual training batches.
        inputs = torch.from_numpy(self.transform_inputs(X))
        labels = self._prepare_labels(y)
        val_inputs = torch.from_numpy(self.transform_inputs(X_val)).to(self.device) if X_val is not None else None
        val_labels = self._prepare_labels(y_val).to(self.device) if y_val is not None else None
        best, patience = float('inf'), 0
        for epoch in tqdm(range(self.epochs), disable=not verbose, desc='TFA epochs'):
            self.model.train()
            totals = dict.fromkeys(['total', 'prediction', 'reconstruction', 'smoothness', 'orthogonality'], 0.)
            for batch in torch.randperm(len(inputs)).split(self.batch_size):
                self.optimizer.zero_grad(set_to_none=True)
                loss, parts = self.model.compute_loss(inputs[batch].to(self.device), labels[batch].to(self.device),
                                                     alpha=self.alpha, beta=self.beta, gamma=self.gamma)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite TFA loss at epoch {epoch + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
                self.optimizer.step()
                for key in totals:
                    totals[key] += parts[key] * len(batch) / len(inputs)
            history = dict(epoch=epoch + 1, train_loss=totals['total'],
                           **{f'train_{k}': v for k, v in totals.items()})
            if val_inputs is not None:
                assert val_labels is not None
                self.model.eval()
                with torch.no_grad():
                    loss, parts = self.model.compute_loss(val_inputs, val_labels, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite validation loss")
                # Same prediction criterion across auxiliary-loss ablations.
                criterion = parts['prediction']
                history.update(val_loss=criterion, **{f'val_{k}': v for k, v in parts.items()})
                scheduler.step(criterion)
                if criterion < best:
                    best, patience = criterion, 0
                    self.best_epoch = epoch + 1
                    self.best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience += 1
            self.training_history.append(history)
            if val_inputs is not None and patience >= self.early_stopping_patience:
                break
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.quantile_boundaries is None:
            raise ValueError("Predict requires a fitted model")
        scaled = self.transform_inputs(X)
        midpoints = (self.quantile_boundaries[:-1] + self.quantile_boundaries[1:]) / 2
        class_values = torch.tensor(midpoints, dtype=torch.float32, device=self.device)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for batch in torch.from_numpy(scaled).split(self.batch_size):
                probs = self.model.predict_proba(batch.to(self.device))
                outputs.append((probs * class_values).sum(-1).cpu().numpy())
        return np.concatenate(outputs)

    def get_params(self) -> Dict:
        return dict(self.config, model_type='TemporalFactorAutoencoder', n_parameters=self.model.count_parameters())

    def save(self, path: str | Path):
        """Save an inference checkpoint; this is not an optimizer-resume snapshot."""
        if self.quantile_boundaries is None:
            raise ValueError("Cannot save an unfitted predictor")
        payload = dict(format_version=1, config=deepcopy(self.config),
                       model_state={k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                       scaler={key: torch.as_tensor(np.asarray(getattr(self.scaler, key)))
                               for key in ['mean_', 'scale_', 'var_', 'n_samples_seen_']},
                       quantile_boundaries=torch.from_numpy(self.quantile_boundaries.copy()),
                       feature_names=self.feature_names, label_semantics='cross-sectional percentile rank',
                       best_epoch=self.best_epoch)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str = 'cpu'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format_version') != 1:
            raise ValueError("Unsupported checkpoint; legacy weight-only files lack preprocessing")
        model = cls(**dict(payload['config'], device=device))
        model.model.load_state_dict(payload['model_state'])
        for key, tensor in payload['scaler'].items():
            setattr(model.scaler, key, tensor.numpy())
        setattr(model.scaler, 'n_features_in_', model.n_pca_factors)
        model.quantile_boundaries = payload['quantile_boundaries'].numpy()
        model.feature_names = payload['feature_names']
        model.best_epoch = payload['best_epoch']
        model.model.eval()
        return model
