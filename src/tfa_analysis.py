"""
Analysis and visualization tools for Temporal Factor Autoencoder.

Provides functions to:
1. Extract and visualize attention weights
2. Analyze factor importance over time
3. Identify market regimes
4. Generate trading signals from attention patterns
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFAAnalyzer:
    """
    Analyzer for Temporal Factor Autoencoder model.
    
    Provides comprehensive analysis and visualization tools.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: Trained TFA model or TFAPredictor
            device: Device for computation
        """
        self.model = model
        self.device = device
        
        # Extract the underlying TFA model if wrapped
        if hasattr(model, 'model'):
            self.tfa_model = model.model
        else:
            self.tfa_model = model
        
        self.tfa_model.eval()
    
    def extract_factor_weights(
        self,
        X: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        assets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract dynamic factor weights for all samples.
        
        Args:
            X: (n_samples, seq_len, n_features)
            dates: Date index for samples
            assets: Asset names
        
        Returns:
            DataFrame with columns: [date, asset, month_offset, factor, weight]
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            weights = self.tfa_model.get_factor_weights(X_tensor)
        
        weights_np = weights.cpu().numpy()  # (n_samples, seq_len, n_features)
        
        # Convert to long format DataFrame
        records = []
        for i in range(len(weights_np)):
            for t in range(weights_np.shape[1]):
                for f in range(weights_np.shape[2]):
                    records.append({
                        'sample_idx': i,
                        'date': dates[i] if dates is not None else i,
                        'asset': assets[i] if assets is not None else f'asset_{i}',
                        'month_offset': t - weights_np.shape[1],  # Negative offset
                        'factor': f'PC{f+1}',
                        'weight': weights_np[i, t, f]
                    })
        
        return pd.DataFrame(records)
    
    def plot_average_attention_pattern(
        self,
        weights_df: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot average attention pattern across time and factors.
        
        Args:
            weights_df: DataFrame from extract_factor_weights
            save_path: Path to save figure
            figsize: Figure size
        """
        # Aggregate weights
        avg_weights = weights_df.groupby(['month_offset', 'factor'])['weight'].mean().unstack()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Heatmap
        sns.heatmap(
            avg_weights.T,
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            ax=axes[0],
            vmin=0,
            vmax=avg_weights.values.max()
        )
        axes[0].set_title('Average Factor Weights Across Time', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Months Ago')
        axes[0].set_ylabel('PCA Factor')
        
        # Line plot
        for factor in avg_weights.columns:
            axes[1].plot(avg_weights.index, avg_weights[factor], 
                        label=factor, linewidth=2, alpha=0.7)
        
        axes[1].set_title('Temporal Attention Decay', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Months Ago')
        axes[1].set_ylabel('Average Weight')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention pattern to {save_path}")
        
        plt.show()
    
    def analyze_regime_patterns(
        self,
        weights_df: pd.DataFrame,
        market_states: pd.Series,
        save_path: Optional[str] = None
    ):
        """
        Analyze attention patterns in different market regimes.
        
        Args:
            weights_df: DataFrame from extract_factor_weights
            market_states: Series with index=dates, values=regime labels
            save_path: Path to save figure
        """
        # Merge market states
        weights_with_regime = weights_df.merge(
            market_states.rename('regime').reset_index(),
            left_on='date',
            right_on='index',
            how='left'
        )
        
        # Get unique regimes
        regimes = weights_with_regime['regime'].dropna().unique()
        
        fig, axes = plt.subplots(1, len(regimes), figsize=(6*len(regimes), 5))
        if len(regimes) == 1:
            axes = [axes]
        
        for i, regime in enumerate(regimes):
            regime_data = weights_with_regime[weights_with_regime['regime'] == regime]
            avg_weights = regime_data.groupby(['month_offset', 'factor'])['weight'].mean().unstack()
            
            sns.heatmap(
                avg_weights.T,
                cmap='YlOrRd',
                cbar_kws={'label': 'Weight'},
                ax=axes[i],
                vmin=0
            )
            axes[i].set_title(f'Regime: {regime}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Months Ago')
            axes[i].set_ylabel('PCA Factor')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved regime analysis to {save_path}")
        
        plt.show()
    
    def plot_factor_importance_evolution(
        self,
        weights_df: pd.DataFrame,
        window: int = 12,
        save_path: Optional[str] = None
    ):
        """
        Plot how factor importance evolves over calendar time.
        
        Args:
            weights_df: DataFrame from extract_factor_weights
            window: Rolling window for smoothing
            save_path: Path to save figure
        """
        # Focus on most recent month (month_offset = -1)
        recent_weights = weights_df[weights_df['month_offset'] == -1].copy()
        
        # Pivot and compute rolling average
        importance_ts = recent_weights.pivot_table(
            index='date',
            columns='factor',
            values='weight',
            aggfunc='mean'
        ).sort_index()
        
        # Smooth with rolling window
        importance_smooth = importance_ts.rolling(window=window, min_periods=1).mean()
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for factor in importance_smooth.columns:
            ax.plot(importance_smooth.index, importance_smooth[factor],
                   label=factor, linewidth=2, alpha=0.7)
        
        ax.set_title('Factor Importance Evolution Over Time', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Weight (Recent Month)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved factor evolution to {save_path}")
        
        plt.show()
    
    def identify_attention_signals(
        self,
        weights_df: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Identify trading signals from attention patterns.
        
        Args:
            weights_df: DataFrame from extract_factor_weights
            threshold: Threshold for "high attention"
        
        Returns:
            DataFrame with signals
        """
        # Focus on recent 3 months
        recent = weights_df[weights_df['month_offset'].isin([-1, -2, -3])]
        
        # Aggregate by sample and factor
        factor_attention = recent.groupby(['sample_idx', 'factor'])['weight'].sum()
        factor_attention = factor_attention.reset_index()
        
        # Identify high-attention factors
        signals = []
        for idx in factor_attention['sample_idx'].unique():
            sample_weights = factor_attention[factor_attention['sample_idx'] == idx]
            total_weight = sample_weights['weight'].sum()
            
            # Concentration in top factors
            top3_weight = sample_weights.nlargest(3, 'weight')['weight'].sum()
            concentration = top3_weight / total_weight if total_weight > 0 else 0
            
            # Momentum signal: high weight on recent months
            momentum_signal = 1 if concentration > threshold else 0
            
            signals.append({
                'sample_idx': idx,
                'concentration': concentration,
                'momentum_signal': momentum_signal,
                'top_factor': sample_weights.nlargest(1, 'weight')['factor'].iloc[0]
            })
        
        return pd.DataFrame(signals)
    
    def analyze_latent_factors(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze learned latent factors.
        
        Args:
            X: Input sequences
            y: Actual returns
            dates: Date index
        
        Returns:
            DataFrame with latent factors
            Dict with correlation analysis
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            latent = self.tfa_model.get_latent_factors(X_tensor)
        
        latent_np = latent.cpu().numpy()
        
        # Create DataFrame
        latent_df = pd.DataFrame(
            latent_np,
            columns=[f'Latent_{i+1}' for i in range(latent_np.shape[1])]
        )
        
        if dates is not None:
            latent_df['date'] = dates
        
        latent_df['return'] = y
        
        # Correlation analysis
        correlations = {}
        for col in latent_df.columns:
            if col.startswith('Latent'):
                corr = np.corrcoef(latent_df[col], latent_df['return'])[0, 1]
                correlations[col] = corr
        
        # Sort by absolute correlation
        correlations = dict(sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return latent_df, correlations
    
    def plot_latent_factor_analysis(
        self,
        latent_df: pd.DataFrame,
        correlations: Dict,
        save_path: Optional[str] = None
    ):
        """Plot latent factor analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Correlation bar plot
        factors = list(correlations.keys())
        corrs = list(correlations.values())
        
        axes[0, 0].barh(factors, corrs, color='steelblue', alpha=0.7)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Latent Factor Correlations with Returns', fontweight='bold')
        axes[0, 0].set_xlabel('Correlation')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # 2. Correlation matrix of latent factors
        latent_cols = [col for col in latent_df.columns if col.startswith('Latent')]
        corr_matrix = latent_df[latent_cols].corr()
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            ax=axes[0, 1],
            cbar_kws={'label': 'Correlation'}
        )
        axes[0, 1].set_title('Latent Factor Correlation Matrix', fontweight='bold')
        
        # 3. Top factor vs returns
        top_factor = factors[0]
        axes[1, 0].scatter(
            latent_df[top_factor],
            latent_df['return'],
            alpha=0.3,
            s=10
        )
        axes[1, 0].set_title(f'{top_factor} vs Returns (Corr={corrs[0]:.3f})', 
                            fontweight='bold')
        axes[1, 0].set_xlabel(top_factor)
        axes[1, 0].set_ylabel('Return')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Factor distributions
        latent_df[latent_cols].boxplot(ax=axes[1, 1])
        axes[1, 1].set_title('Latent Factor Distributions', fontweight='bold')
        axes[1, 1].set_xlabel('Factor')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved latent factor analysis to {save_path}")
        
        plt.show()
    
    def generate_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        assets: Optional[List[str]] = None,
        output_dir: str = 'tfa_analysis'
    ):
        """
        Generate comprehensive analysis report.
        
        Args:
            X: Input sequences
            y: Actual returns
            dates: Date index
            assets: Asset names
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        logger.info("Generating TFA analysis report...")
        
        # 1. Extract factor weights
        logger.info("1. Extracting factor weights...")
        weights_df = self.extract_factor_weights(X, dates, assets)
        weights_df.to_csv(output_path / 'factor_weights.csv', index=False)
        
        # 2. Plot average attention
        logger.info("2. Plotting attention patterns...")
        self.plot_average_attention_pattern(
            weights_df,
            save_path=output_path / 'attention_pattern.png'
        )
        
        # 3. Factor evolution
        logger.info("3. Analyzing factor evolution...")
        self.plot_factor_importance_evolution(
            weights_df,
            save_path=output_path / 'factor_evolution.png'
        )
        
        # 4. Latent factors
        logger.info("4. Analyzing latent factors...")
        latent_df, correlations = self.analyze_latent_factors(X, y, dates)
        latent_df.to_csv(output_path / 'latent_factors.csv', index=False)
        
        self.plot_latent_factor_analysis(
            latent_df,
            correlations,
            save_path=output_path / 'latent_analysis.png'
        )
        
        # 5. Trading signals
        logger.info("5. Generating trading signals...")
        signals = self.identify_attention_signals(weights_df)
        signals.to_csv(output_path / 'attention_signals.csv', index=False)
        
        logger.info(f"✅ Report saved to {output_path}")
        
        return {
            'weights': weights_df,
            'latent': latent_df,
            'correlations': correlations,
            'signals': signals
        }

