"""
Utility functions for Alpha-Hunter project.

Includes logging, data preprocessing, and visualization helpers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
import logging
from datetime import datetime

# Setup logging
def setup_logger(
    name: str = 'alpha_hunter',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logger with console and optional file output.
    
    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level
    log_file : str, optional
        Path to log file
        
    Returns
    -------
    logger : logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def check_gpu_availability() -> str:
    """
    Check if GPU is available for PyTorch.

    Supports CUDA (NVIDIA) and MPS (Apple Silicon).

    Returns
    -------
    device : str
        'cuda', 'mps', or 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)} (CUDA)")
            return 'cuda'
        elif torch.backends.mps.is_available():
            print("GPU available: Apple Silicon (MPS)")
            return 'mps'
        else:
            print("GPU not available, using CPU")
            return 'cpu'
    except ImportError:
        print("PyTorch not installed, using CPU")
        return 'cpu'


def load_pca_features(
    pca_path: str | Path,
    returns_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load PCA features and optionally merge with returns.
    
    Parameters
    ----------
    pca_path : str or Path
        Path to PCA feature CSV
    returns_path : str or Path, optional
        Path to returns CSV
        
    Returns
    -------
    df : pd.DataFrame
        Combined DataFrame
    """
    df = pd.read_csv(pca_path)
    df['date'] = pd.to_datetime(df['date'])
    
    if returns_path:
        returns = pd.read_csv(returns_path)
        returns['date'] = pd.to_datetime(returns['date'])
        df = df.merge(returns, on=['date', 'asset'], how='left')
    
    return df


def create_factor_heatmap(
    loadings_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None,
):
    """
    Create heatmap of factor loadings.
    
    Parameters
    ----------
    loadings_df : pd.DataFrame
        Factor loadings (factors x features)
    top_n : int, default=20
        Number of top features to display
    save_path : str, optional
        Path to save figure
    """
    # Select top features by absolute loading
    abs_sum = loadings_df.abs().sum(axis=0).sort_values(ascending=False)
    top_features = abs_sum.head(top_n).index
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        loadings_df[top_features].T,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Loading'},
        linewidths=0.5,
    )
    plt.title(f'Top {top_n} Feature Loadings Across Factors')
    plt.xlabel('Factor')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(
    history: List[dict],
    save_path: Optional[str] = None,
):
    """
    Plot training history (loss curves).
    
    Parameters
    ----------
    history : list of dict
        Training history with 'epoch', 'train_loss', 'val_loss' keys
    save_path : str, optional
        Path to save figure
    """
    df = pd.DataFrame(history)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    
    if 'val_loss' in df.columns:
        ax.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_model_performance(
    results_dict: dict,
    metrics: List[str] = ['IC_mean', 'ICIR', 'LS_sharpe'],
    save_path: Optional[str] = None,
):
    """
    Create bar chart comparing model performance.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to performance stats
    metrics : list of str
        Metrics to compare
    save_path : str, optional
        Path to save figure
    """
    models = list(results_dict.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = [results_dict[model].get(metric, 0) for model in models]
        
        axes[i].bar(models, values, alpha=0.7, color='steelblue')
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Value')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def export_predictions_to_excel(
    predictions_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    stats: dict,
    output_path: str | Path,
):
    """
    Export predictions and results to Excel file.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions DataFrame
    portfolio_df : pd.DataFrame
        Portfolio returns DataFrame
    stats : dict
        Summary statistics
    output_path : str or Path
        Output Excel file path
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Predictions
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Portfolio returns
        portfolio_df.to_excel(writer, sheet_name='Portfolio Returns')
        
        # Statistics
        stats_df = pd.DataFrame([stats]).T
        stats_df.columns = ['Value']
        stats_df.to_excel(writer, sheet_name='Statistics')
    
    print(f"Results exported to {output_path}")


def analyze_prediction_distribution(
    predictions_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Analyze and plot prediction distribution.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions DataFrame
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prediction distribution
    axes[0, 0].hist(predictions_df['prediction'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Prediction Distribution')
    axes[0, 0].set_xlabel('Predicted Return')
    axes[0, 0].set_ylabel('Frequency')
    
    # Actual return distribution
    axes[0, 1].hist(predictions_df['actual_return'].dropna(), bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_title('Actual Return Distribution')
    axes[0, 1].set_xlabel('Actual Return')
    axes[0, 1].set_ylabel('Frequency')
    
    # Scatter plot
    sample = predictions_df.dropna().sample(min(5000, len(predictions_df)))
    axes[1, 0].scatter(sample['prediction'], sample['actual_return'], alpha=0.3, s=10)
    axes[1, 0].set_title('Prediction vs Actual')
    axes[1, 0].set_xlabel('Predicted Return')
    axes[1, 0].set_ylabel('Actual Return')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Decile analysis
    predictions_df['decile'] = predictions_df.groupby('date')['prediction'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    decile_returns = predictions_df.groupby('decile')['actual_return'].mean()
    axes[1, 1].bar(decile_returns.index, decile_returns.values, alpha=0.7, color='green')
    axes[1, 1].set_title('Average Return by Prediction Decile')
    axes[1, 1].set_xlabel('Decile (0=Low, 9=High)')
    axes[1, 1].set_ylabel('Average Actual Return')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_model_complexity(model) -> dict:
    """
    Calculate model complexity metrics.
    
    Parameters
    ----------
    model : BasePredictor
        Model instance
        
    Returns
    -------
    complexity : dict
        Complexity metrics (parameters, size, etc.)
    """
    complexity = {'model_type': type(model).__name__}
    
    try:
        import torch
        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
            n_params = sum(p.numel() for p in model.model.parameters())
            n_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            complexity['n_parameters'] = n_params
            complexity['n_trainable_parameters'] = n_trainable
    except ImportError:
        pass
    
    return complexity


def create_timestamp() -> str:
    """Create timestamp string for file naming."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

