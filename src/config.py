"""
Configuration file for Alpha-Hunter project.

Contains default hyperparameters and settings for all models.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Data loading configuration."""
    pca_path: str = "feature/pca_feature_store.csv"
    returns_path: Optional[str] = None  # Path to returns data if separate
    sequence_length: int = 36  # Number of months in sequence (changed to 36 for TFA)
    forward_fill_limit: int = 3  # Max months to forward fill


@dataclass
class TrainingConfig:
    """Training configuration."""
    train_window: int = 60  # Training window in months
    val_window: int = 12  # Validation window in months
    min_train_months: int = 36  # Minimum training months
    output_dir: str = "results"
    save_models: bool = False
    verbose: bool = True


@dataclass
class TransformerConfig:
    """Transformer model hyperparameters (aligned with Optimized TFA)."""
    d_model: int = 64  # Same as Optimized TFA
    nhead: int = 4  # Same as Optimized TFA
    num_layers: int = 2  # Same as Optimized TFA (encoder layers)
    dim_feedforward: int = 256
    dropout: float = 0.1
    lr: float = 5e-4  # Same as Optimized TFA (was 1e-3)
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 50  # Same as Optimized TFA
    early_stopping_patience: int = 5
    device: str = 'cpu'  # 'cpu' or 'cuda'


@dataclass
class RidgeConfig:
    """Ridge regression hyperparameters."""
    alpha: float = 1.0


@dataclass
class RandomForestConfig:
    """Random Forest hyperparameters."""
    n_estimators: int = 100
    max_depth: Optional[int] = 10
    min_samples_split: int = 10
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class MLPConfig:
    """MLP hyperparameters."""
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 50
    device: str = 'cpu'


@dataclass
class TFAConfig:
    """Temporal Factor Autoencoder hyperparameters."""
    n_pca_factors: int = 11  # Number of PCA components
    seq_len: int = 36  # Sequence length (3 years)
    d_model: int = 128  # Model dimension
    n_heads: int = 8  # Number of attention heads
    n_encoder_layers: int = 4  # Number of encoder layers
    n_decoder_layers: int = 2  # Number of decoder layers
    n_latent_factors: int = 5  # Number of learned latent factors
    dropout: float = 0.1  # Dropout rate
    n_classes: int = 5  # Number of return quantiles
    lr: float = 1e-3  # Learning rate
    weight_decay: float = 1e-4  # L2 regularization
    batch_size: int = 128  # Batch size
    epochs: int = 50  # Maximum epochs
    early_stopping_patience: int = 5  # Early stopping patience
    alpha: float = 0.1  # Reconstruction loss weight
    beta: float = 0.05  # Smoothness loss weight
    gamma: float = 0.01  # Orthogonality loss weight
    device: str = 'cpu'  # 'cpu' or 'cuda'


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    long_pct: float = 0.1  # Top 10% for long
    short_pct: float = 0.1  # Bottom 10% for short
    transaction_cost: float = 0.003  # 30 bps per side
    weighting: str = 'equal'  # 'equal' or 'value'
    periods_per_year: int = 12  # Monthly data
    risk_free_rate: float = 0.0


class Config:
    """Master configuration class."""
    
    def __init__(self):
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.transformer = TransformerConfig()
        self.tfa = TFAConfig()  # NEW: Temporal Factor Autoencoder
        self.ridge = RidgeConfig()
        self.random_forest = RandomForestConfig()
        self.mlp = MLPConfig()
        self.evaluation = EvaluationConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        config = cls()
        
        for section, params in config_dict.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in params.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        return config
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'transformer': self.transformer.__dict__,
            'tfa': self.tfa.__dict__,
            'ridge': self.ridge.__dict__,
            'random_forest': self.random_forest.__dict__,
            'mlp': self.mlp.__dict__,
            'evaluation': self.evaluation.__dict__,
        }
    
    def save(self, path: str | Path):
        """Save config to JSON file."""
        import json
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path):
        """Load config from JSON file."""
        import json
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configurations for quick access
DEFAULT_CONFIG = Config()

# Hyperparameter search grids
TRANSFORMER_PARAM_GRID = {
    'd_model': [32, 64, 128],
    'nhead': [2, 4, 8],
    'num_layers': [1, 2, 3],
    'dropout': [0.0, 0.1, 0.2],
    'lr': [1e-4, 1e-3, 1e-2],
}

RIDGE_PARAM_GRID = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
}

RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [5, 10, 20],
}

MLP_PARAM_GRID = {
    'hidden_dims': [[128, 64], [256, 128, 64], [512, 256, 128]],
    'dropout': [0.1, 0.2, 0.3],
    'lr': [1e-4, 1e-3, 1e-2],
}

TFA_PARAM_GRID = {
    'd_model': [64, 128, 256],
    'n_heads': [4, 8],
    'n_encoder_layers': [2, 4, 6],
    'n_latent_factors': [3, 5, 8],
    'alpha': [0.05, 0.1, 0.2],  # Reconstruction weight
    'beta': [0.01, 0.05, 0.1],  # Smoothness weight
    'lr': [5e-4, 1e-3, 2e-3],
}

