"""
Performance evaluation metrics for factor investing.

Implements IC, ICIR, Sharpe ratio, turnover, and portfolio backtesting.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Evaluator for prediction and portfolio performance.
    
    Computes:
    - Information Coefficient (IC)
    - IC Information Ratio (ICIR)
    - Portfolio returns (long-short, long-only)
    - Sharpe ratio
    - Maximum drawdown
    - Turnover
    """
    
    def __init__(self):
        pass
    
    def compute_ic(
        self,
        predictions_df: pd.DataFrame,
        method: str = 'spearman',
    ) -> pd.Series:
        """
        Compute Information Coefficient (cross-sectional rank correlation).
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with columns: date, asset, prediction, actual_return
        method : str, default='spearman'
            Correlation method: 'spearman' or 'pearson'
            
        Returns
        -------
        ic_series : pd.Series
            IC for each date, indexed by date
        """
        ic_list = []
        
        for date, group in predictions_df.groupby('date'):
            pred = group['prediction'].values
            actual = group['actual_return'].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(pred) | np.isnan(actual))
            pred = pred[valid_mask]
            actual = actual[valid_mask]
            
            if len(pred) < 2:
                ic = np.nan
            else:
                if method == 'spearman':
                    ic, _ = stats.spearmanr(pred, actual)
                elif method == 'pearson':
                    ic, _ = stats.pearsonr(pred, actual)
                else:
                    raise ValueError(f"Unknown method: {method}")
            
            ic_list.append({'date': date, 'ic': ic})
        
        ic_series = pd.DataFrame(ic_list).set_index('date')['ic']
        return ic_series
    
    def compute_icir(
        self,
        predictions_df: pd.DataFrame,
        method: str = 'spearman',
    ) -> float:
        """
        Compute IC Information Ratio (mean IC / std IC).
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with predictions and returns
        method : str, default='spearman'
            Correlation method
            
        Returns
        -------
        icir : float
            Information ratio of IC
        """
        ic_series = self.compute_ic(predictions_df, method=method)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        
        if ic_std == 0 or np.isnan(ic_std):
            return np.nan
        
        icir = ic_mean / ic_std
        return icir
    
    def compute_portfolio_returns(
        self,
        predictions_df: pd.DataFrame,
        long_pct: float = 0.1,
        short_pct: float = 0.1,
        transaction_cost: float = 0.003,
        weighting: str = 'equal',
    ) -> pd.DataFrame:
        """
        Compute long-short portfolio returns from predictions.
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with predictions and returns
        long_pct : float, default=0.1
            Percentage of stocks to long (top predictions)
        short_pct : float, default=0.1
            Percentage of stocks to short (bottom predictions)
        transaction_cost : float, default=0.003
            Transaction cost per side (30 bps default)
        weighting : str, default='equal'
            Weighting scheme: 'equal' or 'value'
            
        Returns
        -------
        portfolio_df : pd.DataFrame
            DataFrame with columns: date, long_ret, short_ret, ls_ret, ls_ret_net
        """
        portfolio_records = []
        prev_long_assets = set()
        prev_short_assets = set()
        
        for date, group in predictions_df.groupby('date'):
            # Sort by prediction
            group = group.sort_values('prediction', ascending=False)
            
            n_stocks = len(group)
            n_long = max(1, int(n_stocks * long_pct))
            n_short = max(1, int(n_stocks * short_pct))
            
            # Select long and short baskets
            long_basket = group.head(n_long)
            short_basket = group.tail(n_short)
            
            long_assets = set(long_basket['asset'].values)
            short_assets = set(short_basket['asset'].values)
            
            # Equal-weighted returns
            if weighting == 'equal':
                long_ret = long_basket['actual_return'].mean()
                short_ret = short_basket['actual_return'].mean()
            elif weighting == 'value':
                # Placeholder for value weighting (requires market cap data)
                long_ret = long_basket['actual_return'].mean()
                short_ret = short_basket['actual_return'].mean()
            else:
                raise ValueError(f"Unknown weighting: {weighting}")
            
            # Long-short return (gross)
            ls_ret = long_ret - short_ret
            
            # Compute turnover
            long_turnover = len(long_assets - prev_long_assets) / n_long if len(prev_long_assets) > 0 else 1.0
            short_turnover = len(short_assets - prev_short_assets) / n_short if len(prev_short_assets) > 0 else 1.0
            avg_turnover = (long_turnover + short_turnover) / 2
            
            # Net return after transaction costs
            cost = avg_turnover * transaction_cost * 2  # Both sides
            ls_ret_net = ls_ret - cost
            
            portfolio_records.append({
                'date': date,
                'long_ret': long_ret,
                'short_ret': short_ret,
                'ls_ret': ls_ret,
                'ls_ret_net': ls_ret_net,
                'turnover': avg_turnover,
                'n_long': n_long,
                'n_short': n_short,
            })
            
            prev_long_assets = long_assets
            prev_short_assets = short_assets
        
        portfolio_df = pd.DataFrame(portfolio_records).set_index('date')
        return portfolio_df
    
    def compute_sharpe_ratio(
        self,
        returns: pd.Series | pd.DataFrame,
        periods_per_year: int = 12,
        risk_free_rate: float = 0.0,
        column: Optional[str] = None,
    ) -> float:
        """
        Compute annualized Sharpe ratio.
        
        Parameters
        ----------
        returns : pd.Series or pd.DataFrame
            Return series or DataFrame
        periods_per_year : int, default=12
            Number of periods per year (12 for monthly)
        risk_free_rate : float, default=0.0
            Annual risk-free rate
        column : str, optional
            Column name if returns is a DataFrame
            
        Returns
        -------
        sharpe : float
            Annualized Sharpe ratio
        """
        if isinstance(returns, pd.DataFrame):
            if column is None:
                raise ValueError("Must specify column when returns is a DataFrame")
            returns = returns[column]
        
        returns = returns.dropna()
        
        if len(returns) == 0:
            return np.nan
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        mean_return = excess_returns.mean()
        std_return = excess_returns.std()
        
        if std_return == 0 or np.isnan(std_return):
            return np.nan
        
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return sharpe
    
    def compute_max_drawdown(
        self,
        returns: pd.Series | pd.DataFrame,
        column: Optional[str] = None,
    ) -> float:
        """
        Compute maximum drawdown.
        
        Parameters
        ----------
        returns : pd.Series or pd.DataFrame
            Return series
        column : str, optional
            Column name if returns is a DataFrame
            
        Returns
        -------
        max_dd : float
            Maximum drawdown (positive value)
        """
        if isinstance(returns, pd.DataFrame):
            if column is None:
                raise ValueError("Must specify column when returns is a DataFrame")
            returns = returns[column]
        
        returns = returns.dropna()
        
        # Compute cumulative returns
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        return abs(max_dd)
    
    def compute_turnover(
        self,
        predictions_df: pd.DataFrame,
        top_pct: float = 0.1,
    ) -> pd.Series:
        """
        Compute portfolio turnover over time.
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with predictions
        top_pct : float, default=0.1
            Percentage defining the portfolio
            
        Returns
        -------
        turnover_series : pd.Series
            Turnover for each date
        """
        turnover_list = []
        prev_assets = set()
        
        for date, group in predictions_df.groupby('date'):
            group = group.sort_values('prediction', ascending=False)
            n_select = max(1, int(len(group) * top_pct))
            current_assets = set(group.head(n_select)['asset'].values)
            
            if len(prev_assets) > 0:
                turnover = len(current_assets - prev_assets) / n_select
            else:
                turnover = 1.0
            
            turnover_list.append({'date': date, 'turnover': turnover})
            prev_assets = current_assets
        
        turnover_series = pd.DataFrame(turnover_list).set_index('date')['turnover']
        return turnover_series
    
    def compute_summary_statistics(
        self,
        predictions_df: pd.DataFrame,
        portfolio_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive summary statistics.
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            Predictions DataFrame
        portfolio_df : pd.DataFrame, optional
            Portfolio returns DataFrame
            
        Returns
        -------
        stats : dict
            Dictionary of summary statistics
        """
        stats = {}
        
        # IC statistics
        ic_series = self.compute_ic(predictions_df, method='spearman')
        stats['IC_mean'] = ic_series.mean()
        stats['IC_std'] = ic_series.std()
        stats['IC_IR'] = self.compute_icir(predictions_df, method='spearman')
        stats['IC_positive_ratio'] = (ic_series > 0).mean()
        
        # Pearson IC
        ic_pearson = self.compute_ic(predictions_df, method='pearson')
        stats['IC_pearson_mean'] = ic_pearson.mean()
        
        # Portfolio statistics
        if portfolio_df is None:
            portfolio_df = self.compute_portfolio_returns(predictions_df)
        
        stats['LS_mean_return'] = portfolio_df['ls_ret_net'].mean()
        stats['LS_std_return'] = portfolio_df['ls_ret_net'].std()
        stats['LS_sharpe'] = self.compute_sharpe_ratio(portfolio_df, column='ls_ret_net')
        stats['LS_max_drawdown'] = self.compute_max_drawdown(portfolio_df, column='ls_ret_net')
        
        stats['Long_mean_return'] = portfolio_df['long_ret'].mean()
        stats['Long_sharpe'] = self.compute_sharpe_ratio(portfolio_df, column='long_ret')
        
        stats['Short_mean_return'] = portfolio_df['short_ret'].mean()
        
        stats['Avg_turnover'] = portfolio_df['turnover'].mean()
        
        # Win rate
        stats['LS_win_rate'] = (portfolio_df['ls_ret_net'] > 0).mean()
        
        return stats
    
    def print_summary(self, stats: Dict[str, Any]):
        """Pretty print summary statistics."""
        print("\n" + "="*60)
        print("Performance Summary")
        print("="*60)
        
        print("\nInformation Coefficient:")
        print(f"  IC Mean:              {stats['IC_mean']:.4f}")
        print(f"  IC Std:               {stats['IC_std']:.4f}")
        print(f"  ICIR:                 {stats['IC_IR']:.4f}")
        print(f"  IC>0 Ratio:           {stats['IC_positive_ratio']:.2%}")
        
        print("\nLong-Short Portfolio:")
        print(f"  Mean Return (monthly): {stats['LS_mean_return']:.2%}")
        print(f"  Std Dev:              {stats['LS_std_return']:.2%}")
        print(f"  Sharpe Ratio:         {stats['LS_sharpe']:.4f}")
        print(f"  Max Drawdown:         {stats['LS_max_drawdown']:.2%}")
        print(f"  Win Rate:             {stats['LS_win_rate']:.2%}")
        
        print("\nLong-Only Portfolio:")
        print(f"  Mean Return:          {stats['Long_mean_return']:.2%}")
        print(f"  Sharpe Ratio:         {stats['Long_sharpe']:.4f}")
        
        print("\nTurnover:")
        print(f"  Average Turnover:     {stats['Avg_turnover']:.2%}")
        
        print("="*60 + "\n")
    
    def plot_performance(
        self,
        ic_series: pd.Series,
        portfolio_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        """
        Plot performance charts.
        
        Parameters
        ----------
        ic_series : pd.Series
            IC time series
        portfolio_df : pd.DataFrame
            Portfolio returns
        save_path : str, optional
            Path to save figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_theme(style='whitegrid')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. IC over time
        # Ensure index is datetime
        ic_index = ic_series.index
        if not isinstance(ic_index, pd.DatetimeIndex):
            ic_index = pd.to_datetime(ic_index)
        
        axes[0, 0].plot(ic_index, ic_series.values, alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=ic_series.mean(), color='green', linestyle='--', alpha=0.5, label=f'Mean: {ic_series.mean():.3f}')
        axes[0, 0].set_title('Information Coefficient Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('IC')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Format x-axis dates
        from matplotlib.dates import DateFormatter
        axes[0, 0].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        
        # 2. IC distribution
        axes[0, 1].hist(ic_series.dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].axvline(x=ic_series.mean(), color='green', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('IC Distribution')
        axes[0, 1].set_xlabel('IC')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Cumulative returns
        cum_returns = (1 + portfolio_df['ls_ret_net']).cumprod()
        cum_dates = cum_returns.index
        if not isinstance(cum_dates, pd.DatetimeIndex):
            cum_dates = pd.to_datetime(cum_dates)
        
        axes[1, 0].plot(cum_dates, cum_returns.values, linewidth=2)
        axes[1, 0].set_title('Cumulative Long-Short Returns')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Cumulative Return')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Format x-axis dates
        from matplotlib.dates import DateFormatter
        axes[1, 0].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        
        # 4. Monthly returns
        returns = portfolio_df['ls_ret_net'].values
        dates = portfolio_df.index
        
        # Ensure dates are datetime type
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)
        
        # Calculate appropriate bar width based on number of dates
        # Use relative width (days) instead of absolute numeric conversion
        if len(dates) > 1:
            # Calculate average days between dates
            date_diffs = pd.Series(dates).diff().dropna()
            if len(date_diffs) > 0:
                avg_days = date_diffs.mean().total_seconds() / (24 * 3600)  # Convert to days
                bar_width = max(avg_days * 0.6, 1.0)  # 60% of spacing, minimum 1 day
            else:
                bar_width = 20  # Default: 20 days
        else:
            bar_width = 20  # Default: 20 days
        
        # Use different colors for positive and negative returns
        colors = ['green' if r >= 0 else 'red' for r in returns]
        
        axes[1, 1].bar(dates, returns, width=bar_width, alpha=0.7, color=colors, edgecolor='black', linewidth=0.5)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        axes[1, 1].set_title('Monthly Long-Short Returns')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Return')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Format x-axis dates properly
        from matplotlib.dates import DateFormatter
        axes[1, 1].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        
        # Use constrained_layout instead of tight_layout to avoid date overflow issues
        try:
            plt.tight_layout()
        except (OverflowError, ValueError) as e:
            logger.warning(f"tight_layout failed: {e}. Using constrained_layout instead.")
            fig.set_constrained_layout(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.show()

