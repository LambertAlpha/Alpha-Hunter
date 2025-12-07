"""
Rolling window trainer for time-series cross-validation.

Implements walk-forward validation with monthly rebalancing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import pickle
import logging

from .models import BasePredictor
from .data_loader import SequenceDataLoader
from .evaluator import PerformanceEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RollingWindowTrainer:
    """
    Rolling window trainer with walk-forward validation.
    
    Parameters
    ----------
    data_loader : SequenceDataLoader
        Data loader with PCA features and returns
    model_factory : Callable
        Function that returns a new model instance
    train_window : int, default=60
        Number of months in training window
    val_window : int, default=12
        Number of months in validation window (for early stopping)
    min_train_months : int, default=36
        Minimum training months before first prediction
    output_dir : str or Path, default='results'
        Directory to save results
    """
    
    def __init__(
        self,
        data_loader: SequenceDataLoader,
        model_factory: Callable[[], BasePredictor],
        train_window: int = 60,
        val_window: int = 12,
        min_train_months: int = 36,
        output_dir: str | Path = 'results',
    ):
        self.data_loader = data_loader
        self.model_factory = model_factory
        self.train_window = train_window
        self.val_window = val_window
        self.min_train_months = min_train_months
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.results = []
        self.models = {}  # Store trained models by date
        self.evaluator = PerformanceEvaluator()

        # Cache for validation datasets to avoid redundant loading
        self._dataset_cache = {}
    
    def train_and_predict(
        self,
        save_models: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run rolling window training and generate out-of-sample predictions.
        
        Parameters
        ----------
        save_models : bool, default=False
            Whether to save trained models for each date
        verbose : bool, default=True
            Whether to print progress
            
        Returns
        -------
        predictions_df : pd.DataFrame
            DataFrame with columns: date, asset, prediction, actual_return
        """
        dates = self.data_loader.dates
        
        # Determine start date for predictions
        start_idx = max(
            self.min_train_months + self.data_loader.sequence_length,
            self.train_window + self.val_window + self.data_loader.sequence_length,
        )
        
        if start_idx >= len(dates):
            raise ValueError(
                f"Insufficient data: need at least {start_idx} months, "
                f"but only have {len(dates)} months."
            )
        
        logger.info(f"Starting rolling window training from date index {start_idx}")
        logger.info(f"Total prediction dates: {len(dates) - start_idx}")
        
        all_predictions = []
        
        for i in range(start_idx, len(dates)):
            test_date = dates[i]
            
            # Define training window
            train_end_idx = i - 1
            train_start_idx = max(0, train_end_idx - self.train_window)
            train_dates = dates[train_start_idx:train_end_idx]
            
            # Define validation window (last val_window months of training)
            val_start_idx = max(train_start_idx, train_end_idx - self.val_window)
            val_dates = dates[val_start_idx:train_end_idx]
            
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Test Date: {test_date.date()}")
                logger.info(f"Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} months)")
                logger.info(f"Val: {val_dates[0].date()} to {val_dates[-1].date()} ({len(val_dates)} months)")
            
            # Build training data
            try:
                X_train, y_train, assets_train = self._build_dataset(train_dates)
                X_val, y_val, assets_val = self._build_dataset(val_dates)
                
                if verbose:
                    logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
                
                # Additional validation
                if len(X_train) == 0:
                    logger.warning(f"No training samples for {test_date}, skipping")
                    continue
                if len(X_val) == 0:
                    logger.warning(f"No validation samples for {test_date}, skipping")
                    continue
                
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to build training data for {test_date}: {e}")
                else:
                    logger.debug(f"Failed to build training data for {test_date}: {e}")
                continue
            
            # Train model
            model = self.model_factory()
            
            try:
                if hasattr(model, 'fit') and 'X_val' in model.fit.__code__.co_varnames:
                    # Model supports validation (e.g., Transformer)
                    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=verbose)
                else:
                    model.fit(X_train, y_train, verbose=verbose)
                
                if verbose:
                    logger.info("Model training completed")
            
            except Exception as e:
                logger.error(f"Failed to train model for {test_date}: {e}")
                continue
            
            # Generate predictions for test date
            try:
                data_dict = self.data_loader.build_sequences(
                    target_date=test_date,
                    include_target=True,
                    return_dict=True,
                )
                
                X_test = data_dict['X']
                y_test = data_dict.get('y', None)
                assets_test = data_dict['assets']
                
                predictions = model.predict(X_test)
                
                if verbose:
                    logger.info(f"Generated {len(predictions)} predictions")
                
                # Store results
                for asset, pred, actual in zip(assets_test, predictions, y_test if y_test is not None else [None]*len(predictions)):
                    all_predictions.append({
                        'date': test_date,
                        'asset': asset,
                        'prediction': pred,
                        'actual_return': actual,
                    })
                
                # Save model if requested
                if save_models:
                    self.models[test_date] = model
                
            except Exception as e:
                logger.error(f"Failed to generate predictions for {test_date}: {e}")
                continue
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(all_predictions)
        
        if len(predictions_df) > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Rolling window training completed")
            logger.info(f"Total predictions: {len(predictions_df)}")
            logger.info(f"Unique dates: {predictions_df['date'].nunique()}")
            logger.info(f"Unique assets: {predictions_df['asset'].nunique()}")
            
            # Save predictions
            output_path = self.output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictions_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        else:
            logger.warning("No predictions generated!")
        
        return predictions_df
    
    def _build_dataset(
        self,
        dates: list[pd.Timestamp],
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build dataset from multiple dates with caching support.

        Parameters
        ----------
        dates : list of pd.Timestamp
            Dates to include in dataset
        use_cache : bool, default=True
            Whether to use cached data if available

        Returns
        -------
        X : np.ndarray, shape (n_samples, seq_len, n_features)
            Input sequences
        y : np.ndarray, shape (n_samples,)
            Target returns
        assets : np.ndarray, shape (n_samples,)
            Asset identifiers
        """
        # Create cache key from date range
        cache_key = (dates[0], dates[-1])

        # Check cache first
        if use_cache and cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]

        X_list = []
        y_list = []
        assets_list = []

        for date in dates:
            try:
                data_dict = self.data_loader.build_sequences(
                    target_date=date,
                    include_target=True,
                    return_dict=True,
                )

                # Check if we have valid data
                if 'X' not in data_dict or len(data_dict['X']) == 0:
                    logger.debug(f"No valid sequences for date {date}")
                    continue

                X_data = data_dict['X']
                assets_data = data_dict['assets']

                # Check if we have target returns
                if 'y' in data_dict and len(data_dict['y']) > 0:
                    y_data = data_dict['y']
                    # Ensure all arrays have the same length
                    if len(X_data) == len(y_data) == len(assets_data):
                        X_list.append(X_data)
                        y_list.append(y_data)
                        assets_list.append(assets_data)
                    else:
                        logger.warning(
                            f"Mismatched array lengths for date {date}: "
                            f"X={len(X_data)}, y={len(y_data)}, assets={len(assets_data)}"
                        )
                        continue
                else:
                    # No target returns available - skip this date for training
                    logger.debug(f"No target returns for date {date}, skipping")
                    continue

            except Exception as e:
                logger.debug(f"Skipping date {date}: {e}")
                continue

        if len(X_list) == 0:
            raise ValueError(f"No valid sequences found in the specified dates ({len(dates)} dates checked)")

        # Stack arrays
        try:
            X = np.vstack(X_list)
            y = np.concatenate(y_list)
            assets = np.concatenate(assets_list)
        except ValueError as e:
            logger.error(f"Failed to concatenate arrays: {e}")
            logger.error(f"X_list lengths: {[len(x) for x in X_list]}")
            logger.error(f"y_list lengths: {[len(y) for y in y_list]}")
            raise ValueError(f"Failed to concatenate arrays: {e}")

        # Cache the result
        if use_cache:
            self._dataset_cache[cache_key] = (X, y, assets)

        return X, y, assets
    
    def evaluate_predictions(
        self,
        predictions_df: pd.DataFrame,
        metrics: list[str] = ['IC', 'ICIR', 'Sharpe', 'Turnover'],
    ) -> Dict[str, Any]:
        """
        Evaluate prediction performance.
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with predictions and actual returns
        metrics : list of str, default=['IC', 'ICIR', 'Sharpe', 'Turnover']
            Metrics to compute
            
        Returns
        -------
        results : dict
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Overall IC and ICIR
        if 'IC' in metrics or 'ICIR' in metrics:
            ic_series = self.evaluator.compute_ic(predictions_df)
            if 'IC' in metrics:
                results['IC_mean'] = ic_series.mean()
                results['IC_std'] = ic_series.std()
            if 'ICIR' in metrics:
                results['ICIR'] = self.evaluator.compute_icir(predictions_df)
        
        # Portfolio performance
        if 'Sharpe' in metrics or 'Turnover' in metrics:
            portfolio_returns = self.evaluator.compute_portfolio_returns(
                predictions_df,
                long_pct=0.1,  # Top 10%
                short_pct=0.1,  # Bottom 10%
            )
            
            if 'Sharpe' in metrics:
                results['Sharpe'] = self.evaluator.compute_sharpe_ratio(portfolio_returns)
            
            if 'Turnover' in metrics:
                turnover = self.evaluator.compute_turnover(predictions_df)
                results['Turnover_mean'] = turnover.mean()
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info("Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save results
        results_path = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")
        
        return results
    
    def save_models(self, filename: str = 'trained_models.pkl'):
        """Save all trained models."""
        save_path = self.output_dir / filename
        with open(save_path, 'wb') as f:
            pickle.dump(self.models, f)
        logger.info(f"Saved {len(self.models)} models to {save_path}")
    
    def load_models(self, filename: str = 'trained_models.pkl'):
        """Load trained models."""
        load_path = self.output_dir / filename
        with open(load_path, 'rb') as f:
            self.models = pickle.load(f)
        logger.info(f"Loaded {len(self.models)} models from {load_path}")

