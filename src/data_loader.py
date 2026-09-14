"""
Data loading and sequence construction for time-series prediction.

Loads PCA features and constructs rolling windows of sequences for model input.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, overload

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceDataLoader:
    """
    Loads PCA feature store and constructs sequences for time-series models.
    
    Parameters
    ----------
    pca_path : str or Path
        Path to PCA feature store CSV file
    sequence_length : int, default=12
        Number of months to include in each sequence
    forward_fill_limit : int, default=3
        Maximum number of months to forward-fill missing data
    """
    
    def __init__(
        self,
        pca_path: str | Path,
        returns_path: Optional[str | Path] = None,
        sequence_length: int = 12,
        forward_fill_limit: int = 3,
    ):
        if not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer")
        if not isinstance(forward_fill_limit, int) or forward_fill_limit < 0:
            raise ValueError("forward_fill_limit must be a nonnegative integer")
        self.pca_path = Path(pca_path)
        self.returns_path = Path(returns_path) if returns_path else None
        self.sequence_length = sequence_length
        self.forward_fill_limit = forward_fill_limit
        
        # Load data
        logger.info(f"Loading PCA features from {self.pca_path}")
        self.df = self._load_and_validate()
        
        # Extract metadata
        self.feature_columns = [col for col in self.df.columns if col.startswith('pca_')]
        self.n_features = len(self.feature_columns)
        self.dates = sorted(self.df['date'].unique())
        self.assets = sorted(self.df['asset'].unique())
        
        logger.info(f"Loaded {len(self.df)} records")
        logger.info(f"Features: {self.n_features}, Dates: {len(self.dates)}, Assets: {len(self.assets)}")
    
    def _load_and_validate(self) -> pd.DataFrame:
        """Load and validate PCA feature store."""
        def read_panel(path):
            frame = pd.read_csv(path, dtype={'asset': str})
            if frame.empty or not {'date', 'asset'}.issubset(frame.columns):
                raise ValueError("Nonempty panel requires date and asset columns")
            if frame[['date', 'asset']].isna().to_numpy().any() or frame.asset.str.strip().eq('').any():
                raise ValueError("Missing date/asset identifiers")
            frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.to_period('M').dt.to_timestamp()
            if frame.date.isna().any():
                raise ValueError('Missing parsed date')
            if frame.duplicated(['date', 'asset']).any():
                raise ValueError("Duplicate asset/month observations")
            return frame

        df = read_panel(self.pca_path)
        features = [col for col in df if col.startswith('pca_')]
        if not features:
            raise ValueError("At least one pca_ feature is required")
        if self.returns_path:
            if 'return' in df:
                raise ValueError("Choose embedded returns or returns_path, not both")
            returns = read_panel(self.returns_path)
            if 'return' not in returns:
                raise ValueError("Returns panel requires a return column")
            df = df.merge(returns[['date', 'asset', 'return']], how='left',
                          on=['date', 'asset'], validate='one_to_one')
        for column in features + (['return'] if 'return' in df else []):
            df[column] = pd.to_numeric(df[column], errors='raise')
            if np.isinf(df[column].to_numpy()).any():
                raise ValueError(f"Infinite values in {column}")
        if 'return' in df and (df['return'] < -1).any():
            raise ValueError("Asset simple returns must be >= -1")
        months = pd.DatetimeIndex(sorted(df.date.unique())).to_period('M').asi8
        if len(months) > 1 and not (np.diff(months) == 1).all():
            raise ValueError("Missing calendar months in the PCA panel")
        return df.sort_values(['date', 'asset']).reset_index(drop=True)

    @overload
    def build_sequences(
        self,
        target_date: pd.Timestamp,
        include_target: bool = True,
        return_dict: Literal[False] = False,
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    @overload
    def build_sequences(
        self,
        target_date: pd.Timestamp,
        include_target: bool = True,
        *,
        return_dict: Literal[True],
    ) -> Dict[str, Any]: ...

    @overload
    def build_sequences(
        self,
        target_date: pd.Timestamp,
        include_target: bool,
        return_dict: Literal[True],
    ) -> Dict[str, Any]: ...

    def build_sequences(
        self,
        target_date: pd.Timestamp,
        include_target: bool = True,
        return_dict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray] | Dict[str, Any]:
        """
        Build sequences ending at target_date (vectorized for performance).

        Parameters
        ----------
        target_date : pd.Timestamp
            The prediction target date
        include_target : bool, default=True
            Whether to include target returns in output
        return_dict : bool, default=False
            Whether to return a dictionary instead of tuple

        Returns
        -------
        If return_dict=False:
            X : np.ndarray, shape (n_samples, sequence_length, n_features)
                Input sequences
            y : np.ndarray, shape (n_samples,)
                Target returns (only if include_target=True)
            assets : np.ndarray, shape (n_samples,)
                Asset identifiers

        If return_dict=True:
            Dictionary with keys 'X', 'y', 'assets', 'date'
        """
        # Get sequence dates
        normalized_date = pd.Timestamp(target_date)
        if not isinstance(normalized_date, pd.Timestamp):
            raise ValueError('Target date must not be NaT')
        target_date = normalized_date.to_period('M').to_timestamp()
        if include_target and 'return' not in self.df:
            raise ValueError("Supervised sequences require return labels")
        target_idx = self.dates.index(target_date)

        if target_idx < self.sequence_length:
            raise ValueError(
                f"Not enough history for target_date {target_date}. "
                f"Need at least {self.sequence_length} months."
            )

        sequence_dates = self.dates[target_idx - self.sequence_length : target_idx]

        # Vectorized approach: pivot all sequence data at once
        sequence_data = self.df[self.df['date'].isin(sequence_dates)]

        # Create multi-index pivot: assets x dates x features
        pivoted = sequence_data.pivot_table(
            index='asset',
            columns='date',
            values=self.feature_columns,
            aggfunc='first'  # In case of duplicates
        )

        # Pandas pivots to (feature, date) columns. Fill each feature along
        # time independently, then explicitly arrange columns as (date, feature).
        # Reindexing also restores dates/features whose observations are all NaN.
        filled_features = {}
        for feature in self.feature_columns:
            columns = pd.MultiIndex.from_product([[feature], sequence_dates])
            values = pivoted.reindex(columns=columns)
            values.columns = sequence_dates
            filled_features[feature] = (values.ffill(axis=1, limit=self.forward_fill_limit)
                                        if self.forward_fill_limit else values)
        filled = pd.concat(filled_features, axis=1).swaplevel(0, 1, axis=1)
        filled = filled.reindex(
            columns=pd.MultiIndex.from_product([sequence_dates, self.feature_columns])
        )

        # Find assets with complete sequences (no NaN after ffill)
        complete_mask = filled.notna().all(axis=1)
        complete_assets = filled[complete_mask].index

        if len(complete_assets) == 0:
            raise ValueError(
                f"No valid sequences found for target_date {target_date}. "
                f"No assets have complete {self.sequence_length}-month history."
            )

        # Get target date data for filtering
        target_df = self.df[self.df['date'] == target_date].set_index('asset')

        if len(target_df) == 0:
            raise ValueError(f"No data available for target_date {target_date}")

        # Filter to assets present in target date
        valid_assets = complete_assets.intersection(target_df.index)

        if len(valid_assets) == 0:
            raise ValueError(
                f"No valid sequences found for target_date {target_date}. "
                f"No complete sequences for assets in target date."
            )

        # Further filter by non-missing returns if needed
        eligible_count = len(valid_assets)
        has_return_column = 'return' in target_df.columns
        if include_target and has_return_column:
            # Only keep assets with valid returns
            valid_returns_mask = target_df.loc[valid_assets, 'return'].notna()
            valid_assets = pd.Index(valid_assets[valid_returns_mask])

            if len(valid_assets) == 0:
                raise ValueError(
                    f"No valid sequences found for target_date {target_date}. "
                    f"All assets have missing return data."
                )

        # Extract sequences for valid assets
        # Reshape from (n_assets, n_dates*n_features) to (n_assets, n_dates, n_features)
        sequences_flat = filled.loc[valid_assets].values
        n_assets = len(valid_assets)
        n_features = len(self.feature_columns)

        # Reshape: columns are organized as [date1_feat1, date1_feat2, ..., date2_feat1, ...]
        # We need to reshape to (n_assets, seq_len, n_features)
        X = sequences_flat.reshape(n_assets, self.sequence_length, n_features)

        assets = valid_assets.to_numpy()

        # Get targets if requested
        targets = None
        returns = None
        if include_target and has_return_column:
            returns = target_df.loc[valid_assets, 'return'].values
            
            # Apply rank-based transformation (cross-sectional ranking per date)
            # Convert returns to percentile ranks [0, 1]
            # This makes the optimization landscape smoother and aligns with IC metric
            ranks = rankdata(returns)  # Ranking: [1, 2, 3, ..., n]
            targets = ranks / len(ranks)  # Normalize to [0, 1]
            
            logger.debug(
                f"Rank-based transformation for {target_date}: "
                f"returns range [{returns.min():.4f}, {returns.max():.4f}] -> "
                f"ranks range [{targets.min():.4f}, {targets.max():.4f}]"
            )

        if return_dict:
            result = {
                'X': X,
                'assets': assets,
                'date': target_date,
                'coverage': {'target_assets': len(target_df), 'complete_histories': eligible_count,
                             'included_assets': len(assets)},
            }
            if targets is not None:
                result['y'] = targets
                # Training labels remain ranks; portfolio evaluation needs the
                # original signed returns in economic units.
                result['raw_returns'] = returns
            return result
        else:
            if targets is not None:
                return X, targets, assets
            else:
                return X, assets
    
    def get_train_test_dates(
        self,
        train_window: int = 60,
        min_train_months: int = 36,
    ) -> list[Tuple[pd.Timestamp, list[pd.Timestamp]]]:
        """
        Generate rolling window train/test date splits.
        
        Parameters
        ----------
        train_window : int, default=60
            Number of months in training window
        min_train_months : int, default=36
            Minimum months required before first prediction
            
        Returns
        -------
        List of (test_date, train_dates) tuples
        """
        splits = []
        
        # Start from min_train_months + sequence_length
        if train_window < min_train_months or min_train_months < 1:
            raise ValueError("train_window must be >= min_train_months >= 1")
        start_idx = min_train_months + self.sequence_length
        
        for i in range(start_idx, len(self.dates)):
            test_date = self.dates[i]
            
            # Training window
            train_start_idx = max(self.sequence_length, i - train_window)
            train_dates = self.dates[train_start_idx:i]
            
            if len(train_dates) >= min_train_months:
                splits.append((test_date, train_dates))
        
        logger.info(f"Generated {len(splits)} rolling window splits")
        return splits

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'n_dates': len(self.dates),
            'n_assets': len(self.assets),
            'n_features': self.n_features,
            'date_range': (self.dates[0], self.dates[-1]),
            'avg_assets_per_date': len(self.df) / len(self.dates),
            'feature_names': self.feature_columns,
        }
        
        if 'return' in self.df.columns:
            returns = self.df['return'].dropna()
            stats['return_stats'] = {
                'mean': returns.mean(),
                'std': returns.std(),
                'min': returns.min(),
                'max': returns.max(),
                'n_valid': len(returns),
                'coverage_pct': 100 * len(returns) / len(self.df),
            }
        else:
            stats['return_stats'] = None
        
        return stats
