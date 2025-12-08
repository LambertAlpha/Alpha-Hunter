"""
Quick script to visualize TFA attention/latent factors for a few dates.

Usage:
    python -m src.plot_tfa_attention --pca_path data/pca/old/pca_feature_store.csv --model_path results/tfa_oldpca
Outputs:
    Saves heatmaps under <model_path>/analysis/*.png
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import SequenceDataLoader
from src.models_tfa import TFAPredictor


def load_model(model_dir: Path):
    # Dummy loader: in current pipeline we don't save state_dict; placeholder for future extension.
    # Here we just return a fresh model with matching shapes to extract weights on a batch.
    raise NotImplementedError("Model weights are not persisted in current pipeline.")


def visualize_weights(weights: torch.Tensor, dates, save_path: Path, title: str):
    """
    weights: (batch, seq_len, n_factors)
    We'll aggregate over batch (mean) to get seq_len x n_factors heatmap.
    """
    w = weights.mean(dim=0).cpu().numpy()  # seq_len x n_factors
    plt.figure(figsize=(8, 6))
    sns.heatmap(w.T, cmap="viridis", cbar_kws={"label": "Attention weight"})
    plt.xlabel("Time (lag index)")
    plt.ylabel("PCA factor")
    plt.title(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_path", type=str, required=True)
    parser.add_argument("--returns_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True, help="Directory containing config/stats; placeholder for weights")
    parser.add_argument("--sample_dates", type=int, default=3, help="Number of tail dates to visualize")
    args = parser.parse_args()

    # Load data
    loader = SequenceDataLoader(
        pca_path=args.pca_path,
        returns_path=args.returns_path,
        sequence_length=36,
        forward_fill_limit=3,
    )
    stats = loader.get_statistics()
    dates = stats["date_range"]
    tail_dates = loader.dates[-args.sample_dates:]

    # Placeholder model (no weights saved): raise for now.
    raise SystemExit("Attention visualization placeholder: current pipeline没有保存TFA权重，需先扩展保存/加载模型。")


if __name__ == "__main__":
    main()
