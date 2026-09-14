"""Walk-forward training with explicit coverage and a bounded monthly cache."""
import inspect
import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data_loader import SequenceDataLoader
from .evaluator import PerformanceEvaluator


class RollingWindowTrainer:
    def __init__(self, data_loader: SequenceDataLoader, model_factory: Callable[..., Any],
                 train_window: int = 60, val_window: int = 12, min_train_months: int = 36,
                 output_dir: str | Path = 'results'):
        for name, value, minimum in [('train_window', train_window, 1), ('val_window', val_window, 0),
                                     ('min_train_months', min_train_months, 1)]:
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if train_window < min_train_months:
            raise ValueError('train_window must be >= min_train_months')
        self.data_loader, self.model_factory = data_loader, model_factory
        self.train_window, self.val_window, self.min_train_months = train_window, val_window, min_train_months
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.last_model = None
        self.last_date = None
        self.run_status: list[dict[str, Any]] = []
        self.evaluator = PerformanceEvaluator()
        self.cache_limit = train_window + val_window + 1
        self._dataset_cache = OrderedDict()

    def train_and_predict(self, save_models: bool = False, verbose: bool = True,
                          max_prediction_dates: Optional[int] = None, prediction_step: int = 1,
                          save_last_model_path: Optional[Path | str] = None,
                          allow_skips: bool = False) -> pd.DataFrame:
        """Fit independent models using disjoint train/validation/target months.

        max_prediction_dates limits the first N eligible calendar months before
        prediction_step is applied. Sparse outputs support ranking diagnostics,
        but must not be compounded as a continuous monthly portfolio.
        """
        if not isinstance(prediction_step, int) or prediction_step < 1:
            raise ValueError('prediction_step must be a positive integer')
        if max_prediction_dates is not None and (not isinstance(max_prediction_dates, int) or max_prediction_dates < 1):
            raise ValueError('max_prediction_dates must be a positive integer')
        dates = self.data_loader.dates
        start = self.train_window + self.val_window + self.data_loader.sequence_length
        if start >= len(dates):
            raise ValueError(f'Need at least {start + 1} monthly dates; found {len(dates)}')
        end = len(dates) if max_prediction_dates is None else min(len(dates), start + max_prediction_dates)
        planned = list(range(start, end, prediction_step))
        self.run_status = [dict(date=str(dates[i].date()), status='planned') for i in planned]
        self.models, self.last_model, self.last_date = {}, None, None
        self._dataset_cache.clear()
        records = []
        self._write_status()
        for index, i in enumerate(tqdm(planned, disable=not verbose, desc='Rolling months')):
            date = dates[i]
            train_dates = dates[i - self.val_window - self.train_window:i - self.val_window]
            val_dates = dates[i - self.val_window:i] if self.val_window else []
            status = self.run_status[index]
            status.update(train_dates=[str(d.date()) for d in train_dates],
                          val_dates=[str(d.date()) for d in val_dates], stage='data')
            try:
                X, y, _ = self._build_dataset(train_dates)
                status['train_missing_returns'] = self._missing_returns(train_dates)
                X_val, y_val, _ = self._build_dataset(val_dates) if val_dates else (None, None, None)
                status['val_missing_returns'] = self._missing_returns(val_dates)
                test = self._date_data(date)
                status.update(train_samples=len(X), validation_samples=len(X_val) if X_val is not None else 0,
                              test_samples=len(test['X']), coverage=test.get('coverage', {}), stage='label_coverage')
                missing = status['coverage'].get('missing_return_assets', [])
                if missing:
                    raise ValueError(f'Missing test returns for {len(missing)} eligible asset(s) at {date.date()}: '
                                     f'{missing[:10]}. Resolve labels upstream; do not select on future availability.')
                status['stage'] = 'fit'
                factory_params = inspect.signature(self.model_factory).parameters
                model = (self.model_factory(rolling_index=index) if 'rolling_index' in factory_params
                         else self.model_factory())
                params = inspect.signature(model.fit).parameters
                variadic = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                arguments = {'verbose': verbose, 'X_val': X_val, 'y_val': y_val}
                model.fit(X, y, **{k: v for k, v in arguments.items() if k in params or variadic})
                status['stage'] = 'predict'
                predicted = np.asarray(model.predict(test['X']), dtype=float)
                if predicted.shape != (len(test['assets']),) or not np.isfinite(predicted).all():
                    raise ValueError('Predictions must be a finite vector with one score per asset')
                frame = pd.DataFrame(dict(date=date, asset=test['assets'], prediction=predicted,
                                          actual_return=test['raw_returns']))
                records.append(frame)
                self.last_model, self.last_date = model, date
                if save_models:
                    self.models[date] = model
                status.update(status='success', stage='complete')
            except Exception as exc:
                status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                if not allow_skips:
                    raise
            finally:
                self._write_status()
        if not records:
            raise RuntimeError('No successful prediction months')
        if save_last_model_path:
            save = getattr(self.last_model, 'save', None)
            if not callable(save):
                raise ValueError('Last model does not implement an inference checkpoint')
            save(save_last_model_path)
        predictions = pd.concat(records, ignore_index=True)
        predictions.to_csv(self.output_dir / 'predictions.csv', index=False)
        return predictions

    def _write_status(self):
        (self.output_dir / 'split_status.json').write_text(json.dumps(self.run_status, indent=2))

    def _date_data(self, date: pd.Timestamp, use_cache: bool = True) -> dict:
        if use_cache and date in self._dataset_cache:
            self._dataset_cache.move_to_end(date)
            return self._dataset_cache[date]
        data = self.data_loader.build_sequences(date, include_target=True, return_dict=True)
        X, y, assets = data['X'], data['y'], data['assets']
        if len(X) == 0 or y.shape != (len(X),) or len(assets) != len(X) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f'Invalid supervised sequences for {date}')
        if use_cache:
            self._dataset_cache[date] = data
            while len(self._dataset_cache) > self.cache_limit:
                self._dataset_cache.popitem(last=False)
        return data

    def _build_dataset(self, dates: list[pd.Timestamp], use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not dates:
            raise ValueError('No dates supplied')
        batches = [self._date_data(d, use_cache) for d in dates]
        X, y, assets = [np.concatenate([b[key] for b in batches]) for key in ['X', 'y', 'assets']]
        return X, y, assets

    def _missing_returns(self, dates) -> dict:
        return {str(d.date()): missing for d in dates
                if (missing := self._date_data(d).get('coverage', {}).get('missing_return_assets', []))}

    def evaluate_predictions(self, predictions_df: pd.DataFrame,
                             metrics: Optional[list[str]] = None) -> dict:
        portfolio = self.evaluator.compute_portfolio_returns(predictions_df)
        summary = self.evaluator.generate_summary_statistics(predictions_df, portfolio)
        if metrics is None:
            return summary
        mapping = {'IC': ['IC_mean', 'IC_std'], 'ICIR': ['IC_IR'],
                   'Sharpe': ['LS_sharpe'], 'Turnover': ['Avg_turnover']}
        return {key: summary[key] for metric in metrics for key in mapping[metric]}

    def save_models(self, filename: str = 'trained_models.pkl'):
        """Trusted local Python archive; prefer TFA inference checkpoints for sharing."""
        with (self.output_dir / filename).open('wb') as stream:
            pickle.dump(self.models, stream)

    def load_models(self, filename: str = 'trained_models.pkl'):
        """Only load archives you trust; pickle can execute code."""
        with (self.output_dir / filename).open('rb') as stream:
            self.models = pickle.load(stream)
