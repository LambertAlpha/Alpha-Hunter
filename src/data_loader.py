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
        self.dates = self.feature_dates.tolist()
        next_month = self.feature_dates[-1] + pd.offsets.MonthBegin()
        if self.return_df is not None:
            last_members = self.feature_df.loc[self.feature_df.date == self.feature_dates[-1], 'asset']
            has_next_labels = self.return_df.date.eq(next_month) & self.return_df.asset.isin(last_members)
            if has_next_labels.any():
                self.dates.append(next_month)
        self.assets = sorted(self.feature_df['asset'].unique())
        
        logger.info(f"Loaded {len(self.df)} records")
        logger.info(f"Features: {self.n_features}, Dates: {len(self.dates)}, Assets: {len(self.assets)}")
    
    def _load_and_validate(self) -> pd.DataFrame:
        """Load and validate PCA feature store."""
        def read_panel(path) -> pd.DataFrame:
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
        # Feature membership and labels have different availability dates. Keep
        # them separate: a delisting return may have no same-month feature row.
        self.feature_df = df.reindex(columns=['date', 'asset', *features])
        self.return_df: Optional[pd.DataFrame] = df.reindex(columns=['date', 'asset', 'return']) if 'return' in df else None
        if self.returns_path:
            if 'return' in df:
                raise ValueError("Choose embedded returns or returns_path, not both")
            returns = read_panel(self.returns_path)
            if 'return' not in returns:
                raise ValueError("Returns panel requires a return column")
            self.return_df = returns.reindex(columns=['date', 'asset', 'return'])
        for column in features:
            self.feature_df[column] = pd.to_numeric(self.feature_df[column], errors='raise')
            if np.isinf(self.feature_df[column].to_numpy()).any():
                raise ValueError(f"Infinite values in {column}")
        self.feature_dates = pd.DatetimeIndex(sorted(self.feature_df.date.unique()))
        months = self.feature_dates.to_period('M').asi8
        if len(months) > 1 and not (np.diff(months) == 1).all():
            raise ValueError("Missing calendar months in the PCA panel")
        df = self.feature_df
        if self.return_df is not None:
            self.return_df['return'] = pd.to_numeric(self.return_df['return'], errors='raise')
            if np.isinf(self.return_df['return'].to_numpy()).any() or (self.return_df['return'] < -1).any():
                raise ValueError("Asset simple returns must be finite or missing and >= -1")
            # The combined view retains all label records for diagnostics;
            # only feature_df is allowed to determine prediction eligibility.
            df = df.merge(self.return_df, how='outer', on=['date', 'asset'], validate='one_to_one')
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
        Build sequences strictly before target_date using past feature membership.

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
        if not isinstance(normalized_date, pd.Timestamp) or pd.isna(normalized_date):
            raise ValueError('Target date must not be NaT')
        target_date = normalized_date.to_period('M').to_timestamp()
        if include_target and self.return_df is None:
            raise ValueError("Supervised sequences require return labels")
        sequence_dates = pd.date_range(end=target_date - pd.offsets.MonthBegin(),
                                       periods=self.sequence_length, freq='MS')
        if not sequence_dates.isin(self.feature_dates).all():
            raise ValueError(
                f"Not enough observed feature history for target_date {target_date}. "
                f"Need the preceding {self.sequence_length} consecutive months."
            )

        # Vectorized approach: pivot all sequence data at once
        sequence_data = self.feature_df[self.feature_df['date'].isin(sequence_dates.tolist())]

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

        # Last observed membership is a documented proxy, not a security master.
        # Never use target-month rows or future label availability as its input.
        universe = pd.Index(sequence_data.loc[sequence_data.date == sequence_dates[-1], 'asset'])
        valid_assets = complete_assets.intersection(universe).sort_values()

        if len(valid_assets) == 0:
            raise ValueError(
                f"No valid sequences found for target_date {target_date}. "
                f"No complete sequences for assets in the last input month."
            )

        eligible_count = len(valid_assets)
        missing_return_assets = []
        target_returns = None
        if include_target:
            assert self.return_df is not None
            target_returns = self.return_df.loc[self.return_df.date == target_date].set_index('asset')['return']
            target_returns = target_returns.reindex(valid_assets)
            valid_returns_mask = target_returns.notna()
            missing_return_assets = pd.Index(valid_assets[~valid_returns_mask]).tolist()
            # Historical supervised fitting can exclude unavailable labels.
            # Rolling test evaluation explicitly rejects any such exclusions.
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
        if target_returns is not None:
            returns = target_returns.loc[valid_assets].to_numpy()
            
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
                'coverage': {'universe_date': str((target_date - pd.offsets.MonthBegin()).date()),
                             'history_universe_assets': len(universe),
                             'complete_histories': eligible_count,
                             'included_assets': len(assets),
                             'missing_return_assets': missing_return_assets},
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
            'avg_assets_per_date': len(self.feature_df) / len(self.feature_dates),
            'n_feature_dates': len(self.feature_dates),
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
