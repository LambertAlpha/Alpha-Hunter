"""Monthly cross-sectional scores and explicit self-financing portfolio accounting."""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def validated_panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'date', 'asset', 'prediction', 'actual_return'}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Nonempty predictions must contain {sorted(required)}")
    frame = frame.copy()
    if frame[list(required)].isna().to_numpy().any():
        raise ValueError("Prediction observations must not contain missing values")
    frame['date'] = pd.to_datetime(frame['date'])
    frame['asset'] = frame['asset'].astype(str)
    if frame.date.isna().any() or frame.asset.str.strip().eq('').any():
        raise ValueError('Missing date/asset identifiers')
    frame[['prediction', 'actual_return']] = frame[['prediction', 'actual_return']].apply(pd.to_numeric, errors='raise')
    if frame.duplicated(['date', 'asset']).any():
        raise ValueError("Duplicate date/asset observations")
    values = frame[['prediction', 'actual_return']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (frame.actual_return < -1).any():
        raise ValueError("Scores/returns must be finite and asset simple returns >= -1")
    return frame.sort_values(['date', 'asset'])


class PerformanceEvaluator:
    def compute_ic(self, predictions_df: pd.DataFrame, method: str = 'spearman') -> pd.Series:
        if method not in {'spearman', 'pearson'}:
            raise ValueError(f"Unknown correlation method: {method}")
        panel = validated_panel(predictions_df)
        values = {}
        for date, group in panel.groupby('date', sort=True):
            pred, actual = group.prediction, group.actual_return
            if len(group) < 2 or pred.nunique() < 2 or actual.nunique() < 2:
                values[date] = np.nan
            else:
                func = stats.spearmanr if method == 'spearman' else stats.pearsonr
                values[date] = float(func(pred, actual)[0])
        return pd.Series(values, name='ic').rename_axis('date')

    def compute_icir(self, predictions_df: pd.DataFrame, method: str = 'spearman') -> float:
        ic = self.compute_ic(predictions_df, method).dropna()
        std = ic.std()
        return float(ic.mean() / std) if std > 0 else float('nan')

    def compute_portfolio_returns(
        self, predictions_df: pd.DataFrame, long_pct: float = 0.1,
        short_pct: float = 0.1, transaction_cost: float = 0.003,
        weighting: str = 'equal', long_weight: float = 0.5,
        short_weight: float = 0.5, min_ls_return: Optional[float] = None,
    ) -> pd.DataFrame:
        """Rebalance signed risky weights at each month start, then earn returns.

        Cost is one-way rate times L1 traded notional, including initial entry.
        Turnover is half that notional (cash excluded). Previous weights drift
        with asset returns and net NAV; residual cash finances trades/costs.
        Cash earns zero. Borrow fees, market impact and terminal liquidation are
        excluded. Weights are fractions of NAV before the current rebalance.
        """
        if weighting != 'equal':
            raise ValueError("Only equal weighting is implemented")
        if min_ls_return is not None:
            raise ValueError("Return clipping is unsupported; insolvency must be reported")
        params = [long_pct, short_pct, transaction_cost, long_weight, short_weight]
        if not np.isfinite(params).all() or min(params) < 0:
            raise ValueError("Portfolio parameters must be finite and nonnegative")
        if long_pct > 1 or short_pct > 1 or long_weight + short_weight == 0:
            raise ValueError("Invalid portfolio fractions or zero gross exposure")
        if (long_weight > 0 and long_pct == 0) or (short_weight > 0 and short_pct == 0):
            raise ValueError("An enabled leg requires a positive selection fraction")
        panel = validated_panel(predictions_df)
        dates = pd.DatetimeIndex(panel.date.unique())
        periods = dates.to_period('M').asi8
        if len(periods) > 1 and not (np.diff(periods) == 1).all():
            raise ValueError("Monthly portfolio evaluation requires consecutive months")
        previous = pd.Series(dtype=float)
        records = []
        for date, group in panel.groupby('date', sort=True):
            group = group.sort_values(['prediction', 'asset'], ascending=[False, True])
            n_long = max(1, int(len(group) * long_pct)) if long_weight else 0
            n_short = max(1, int(len(group) * short_pct)) if short_weight else 0
            if n_long + n_short > len(group):
                raise ValueError(f"{date}: too few assets for disjoint long/short baskets")
            longs = group.head(n_long)
            shorts = group.tail(n_short) if n_short else group.iloc[:0]
            target = pd.Series(0., index=group.asset)
            target.loc[longs.asset] = long_weight / n_long if n_long else 0.
            target.loc[shorts.asset] = -short_weight / n_short if n_short else 0.
            trades = target.subtract(previous, fill_value=0)
            notional = float(trades.abs().sum())
            cost = transaction_cost * notional
            long_ret = float(longs.actual_return.mean()) if n_long else 0.
            short_ret = float(shorts.actual_return.mean()) if n_short else 0.
            gross = long_weight * long_ret - short_weight * short_ret
            net = gross - cost
            if net <= -1:
                raise ValueError(f"{date}: portfolio insolvent (net simple return {net})")
            asset_returns = group.set_index('asset').actual_return
            previous = target * (1 + asset_returns) / (1 + net)
            records.append(dict(date=date, long_ret=long_ret, short_ret=short_ret,
                                ls_ret=gross, ls_ret_net=net, turnover=notional / 2,
                                traded_notional=notional, transaction_cost=cost,
                                n_long=n_long, n_short=n_short))
        return pd.DataFrame(records).set_index('date')

    @staticmethod
    def _returns(returns: pd.Series | pd.DataFrame, column: Optional[str]) -> pd.Series:
        if isinstance(returns, pd.DataFrame):
            if column is None:
                raise ValueError("Must specify column for a returns DataFrame")
            returns = returns[column]
        if not np.isfinite(returns.to_numpy(dtype=float)).all():
            raise ValueError("Returns contain nonfinite observations")
        return pd.Series(returns, dtype=float)

    def compute_sharpe_ratio(self, returns: pd.Series | pd.DataFrame,
                             periods_per_year: int = 12, risk_free_rate: float = 0.,
                             column: Optional[str] = None) -> float:
        returns = self._returns(returns, column)
        if periods_per_year <= 0 or not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
            raise ValueError("Invalid annualization or risk-free rate")
        excess = returns - ((1 + risk_free_rate) ** (1 / periods_per_year) - 1)
        std = excess.std()
        return float(excess.mean() / std * np.sqrt(periods_per_year)) if std > 0 else float('nan')

    def compute_max_drawdown(self, returns: pd.Series | pd.DataFrame,
                             column: Optional[str] = None) -> float:
        returns = self._returns(returns, column)
        if (returns < -1).any():
            raise ValueError("Simple portfolio returns cannot be below -100%")
        wealth = (1 + returns).cumprod()
        high = wealth.cummax().clip(lower=1.)
        return float((1 - wealth / high).max())

    def compute_turnover(self, predictions_df: pd.DataFrame, top_pct: float = .1) -> pd.Series:
        return self.compute_portfolio_returns(predictions_df, long_pct=top_pct,
            short_pct=0, short_weight=0, long_weight=1, transaction_cost=0).turnover

    def generate_summary_statistics(self, predictions_df: pd.DataFrame,
                                    portfolio_df: Optional[pd.DataFrame] = None,
                                    periods_per_year: int = 12,
                                    risk_free_rate: float = 0.) -> Dict[str, Any]:
        ic = self.compute_ic(predictions_df).dropna()
        portfolio = self.compute_portfolio_returns(predictions_df) if portfolio_df is None else portfolio_df
        def sharpe(col):
            return self.compute_sharpe_ratio(portfolio, periods_per_year, risk_free_rate, col)
        return dict(IC_mean=ic.mean(), IC_std=ic.std(), IC_IR=self.compute_icir(predictions_df),
                    IC_positive_ratio=(ic > 0).mean(), IC_valid_months=len(ic),
                    prediction_months=predictions_df.date.nunique(), prediction_rows=len(predictions_df),
                    IC_pearson_mean=self.compute_ic(predictions_df, 'pearson').mean(),
                    LS_mean_return=portfolio.ls_ret_net.mean(), LS_std_return=portfolio.ls_ret_net.std(),
                    LS_sharpe=sharpe('ls_ret_net'), LS_max_drawdown=self.compute_max_drawdown(portfolio, 'ls_ret_net'),
                    Long_mean_return=portfolio.long_ret.mean(), Long_sharpe=sharpe('long_ret'),
                    Short_mean_return=portfolio.short_ret.mean(), Avg_turnover=portfolio.turnover.mean(),
                    Avg_transaction_cost=portfolio.transaction_cost.mean(), LS_win_rate=(portfolio.ls_ret_net > 0).mean())

    def print_summary(self, stats: Dict[str, Any]):
        for name, value in stats.items():
            print(f"{name}: {value:.6g}")

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
            fig.set_layout_engine('constrained')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.close(fig)
