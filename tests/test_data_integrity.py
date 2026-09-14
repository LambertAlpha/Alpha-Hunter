"""Regression tests for chronological inputs and economic return evaluation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_loader import SequenceDataLoader
from src.models import BasePredictor
from src.trainer import RollingWindowTrainer


class RecordingPredictor(BasePredictor):
    def fit(self, X, y, X_val=None, y_val=None, verbose=False, **kwargs):
        return self

    def predict(self, X):
        return np.arange(len(X), dtype=float)


class DataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frame = pd.DataFrame([
            {'date': date, 'asset': asset, 'pca_0': i + 1,
             'pca_1': 101 + i, 'return': ret}
            for i, date in enumerate(pd.date_range('2020-01-01', periods=10, freq='MS'))
            for asset, ret in [('A', -0.1), ('B', 0.1)]
        ])

    def loader(self, frame=None):
        path = self.root / 'features.csv'
        (self.frame if frame is None else frame).to_csv(path, index=False)
        return SequenceDataLoader(path, sequence_length=2)

    def test_sequences_keep_time_and_feature_axes(self):
        loader = self.loader()
        batch = loader.build_sequences(loader.dates[2], return_dict=True)
        np.testing.assert_array_equal(batch['X'][0], [[1, 101], [2, 102]])

    def test_forward_fill_does_not_borrow_another_feature(self):
        frame = self.frame.copy()
        frame.loc[(frame.asset == 'A') & (frame.date == frame.date.min()), 'pca_1'] = np.nan
        loader = self.loader(frame)
        batch = loader.build_sequences(loader.dates[2], return_dict=True)
        self.assertEqual(batch['assets'].tolist(), ['B'])

    def test_raw_returns_are_separate_from_rank_targets(self):
        loader = self.loader()
        batch = loader.build_sequences(loader.dates[2], return_dict=True)
        np.testing.assert_allclose(batch['y'], [0.5, 1.0])
        np.testing.assert_allclose(batch['raw_returns'], [-0.1, 0.1])

    def test_training_validation_and_test_are_chronological(self):
        loader = self.loader()
        trainer = RollingWindowTrainer(loader, RecordingPredictor, train_window=2,
                                       val_window=2, min_train_months=2,
                                       output_dir=self.root / 'results')
        with patch.object(trainer, '_build_dataset', wraps=trainer._build_dataset) as build:
            predictions = trainer.train_and_predict(verbose=False, max_prediction_dates=1)
        train_dates, val_dates = [call.args[0] for call in build.call_args_list]
        self.assertEqual(len(train_dates), 2)
        self.assertEqual(len(val_dates), 2)
        self.assertLess(max(train_dates), min(val_dates))
        self.assertLess(max(val_dates), predictions.date.min())

    def test_prediction_export_contains_economic_returns(self):
        loader = self.loader()
        trainer = RollingWindowTrainer(loader, RecordingPredictor, train_window=2,
                                       val_window=2, min_train_months=2,
                                       output_dir=self.root / 'results')
        predictions = trainer.train_and_predict(verbose=False, max_prediction_dates=1)
        self.assertEqual(len(predictions), 2)
        np.testing.assert_allclose(predictions.sort_values('asset').actual_return, [-0.1, 0.1])


if __name__ == '__main__':
    unittest.main()
