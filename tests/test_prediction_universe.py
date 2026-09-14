"""Future records must not decide which assets could have been predicted."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.cli import run_experiment
from src.config import Config
from src.data_loader import SequenceDataLoader


class PredictionUniverseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dates = pd.date_range('2020-01-01', periods=8, freq='MS')
        self.frame = pd.DataFrame([
            {'date': date, 'asset': asset, 'pca_0': i + offset, 'return': ret}
            for i, date in enumerate(self.dates)
            for asset, offset, ret in [('001', 0., .1), ('002', 1., -.1)]
        ])

    def loader(self, frame, returns=None):
        path = self.root / 'features.csv'
        frame.to_csv(path, index=False)
        returns_path = None
        if returns is not None:
            returns_path = self.root / 'returns.csv'
            returns.to_csv(returns_path, index=False)
        return SequenceDataLoader(path, returns_path=returns_path, sequence_length=2)

    def test_target_month_membership_does_not_select_predictions(self):
        target = self.dates[5]
        original = self.loader(self.frame).build_sequences(target, include_target=False, return_dict=True)
        changed = self.frame[~((self.frame.date >= target) & (self.frame.asset == '002'))]
        actual = self.loader(changed).build_sequences(target, include_target=False, return_dict=True)
        np.testing.assert_array_equal(actual['assets'], original['assets'])
        np.testing.assert_array_equal(actual['X'], original['X'])

    def test_predict_next_month_without_placeholder_or_labels(self):
        features = self.frame.drop(columns='return')
        batch = self.loader(features).build_sequences(self.dates[-1] + pd.offsets.MonthBegin(),
                                                       include_target=False, return_dict=True)
        self.assertEqual(batch['assets'].tolist(), ['001', '002'])
        np.testing.assert_array_equal(batch['X'][0, :, 0], [6., 7.])
        self.assertNotIn('y', batch)

    def test_inference_cannot_jump_over_an_unobserved_month(self):
        with self.assertRaises(ValueError):
            self.loader(self.frame).build_sequences(self.dates[-1] + pd.DateOffset(months=2), include_target=False)

    def test_future_reappearance_cannot_reactivate_a_missing_asset(self):
        target = self.dates[5]
        features = self.frame[~((self.frame.date == self.dates[4]) & (self.frame.asset == '002'))]
        batch = self.loader(features).build_sequences(target, include_target=False, return_dict=True)
        self.assertEqual(batch['assets'].tolist(), ['001'])

    def test_separate_delisting_return_survives_missing_target_features(self):
        target = self.dates[5]
        features = self.frame.drop(columns='return')
        features = features[~((features.date == target) & (features.asset == '002'))]
        returns = self.frame[['date', 'asset', 'return']].copy()
        returns.loc[(returns.date == target) & (returns.asset == '002'), 'return'] = -1.
        batch = self.loader(features, returns).build_sequences(target, return_dict=True)
        self.assertEqual(batch['assets'].tolist(), ['001', '002'])
        np.testing.assert_array_equal(batch['raw_returns'], [.1, -1.])

    def test_labels_can_extend_one_month_past_the_feature_store(self):
        features = self.frame[self.frame.date < self.dates[-1]].drop(columns='return')
        loader = self.loader(features, self.frame[['date', 'asset', 'return']])
        batch = loader.build_sequences(self.dates[-1], return_dict=True)
        self.assertEqual(batch['assets'].tolist(), ['001', '002'])

    def test_unrelated_label_dates_do_not_change_the_rolling_calendar(self):
        returns = pd.concat([self.frame[['date', 'asset', 'return']], pd.DataFrame([
            {'date': pd.Timestamp('1990-01-01'), 'asset': 'other', 'return': .1},
            {'date': self.dates[-1] + pd.offsets.MonthBegin(), 'asset': 'other', 'return': .1},
            {'date': pd.Timestamp('2030-01-01'), 'asset': 'other', 'return': .1},
        ])])
        loader = self.loader(self.frame.drop(columns='return'), returns)
        self.assertEqual(loader.dates, self.dates.tolist())

    def test_missing_test_labels_fail_instead_of_changing_the_basket(self):
        frame = self.frame.copy()
        frame.loc[(frame.date == self.dates[5]) & (frame.asset == '002'), 'return'] = np.nan
        self.loader(frame)
        config = Config.from_dict({'data': {'pca_path': str(self.root / 'features.csv'), 'sequence_length': 2},
                                  'training': {'train_window': 2, 'min_train_months': 2, 'val_window': 1,
                                               'verbose': False, 'max_prediction_dates': 1},
                                  'evaluation': {'long_weight': 1., 'short_weight': 0., 'short_pct': 0.}})
        with self.assertRaisesRegex(ValueError, 'Missing test returns'):
            run_experiment('ridge', config, self.root / 'run')
        status = json.loads((self.root / 'run' / 'split_status.json').read_text())[0]
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['coverage']['missing_return_assets'], ['002'])
        self.assertFalse((self.root / 'run' / 'portfolio.csv').exists())
