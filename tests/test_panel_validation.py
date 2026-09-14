"""Validate monthly panel semantics before any model is fitted."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import SequenceDataLoader


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'panel.csv'
        self.df = pd.DataFrame([
            dict(date=d, asset=a, pca_0=float(i), return_=r)
            for i, d in enumerate(pd.date_range('2020-01-01', periods=5, freq='MS'))
            for a, r in [('001', -.1), ('002', .1)]
        ]).rename(columns={'return_': 'return'})

    def load(self, df=None, **kwargs):
        (self.df if df is None else df).to_csv(self.path, index=False)
        return SequenceDataLoader(self.path, sequence_length=2, **kwargs)

    def test_duplicate_asset_month_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load(pd.concat([self.df, self.df.iloc[:1]]))

    def test_missing_calendar_month_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load(self.df[self.df.date != self.df.date.unique()[1]])

    def test_infinite_feature_is_rejected(self):
        self.df.loc[0, 'pca_0'] = np.inf
        with self.assertRaises(ValueError):
            self.load()

    def test_zero_fill_limit_and_identifiers(self):
        loader = self.load(forward_fill_limit=0)
        batch = loader.build_sequences(loader.dates[2], return_dict=True)
        self.assertEqual(batch['assets'].tolist(), ['001', '002'])

    def test_inference_without_labels_is_explicit(self):
        loader = self.load(self.df.drop(columns='return'))
        self.assertEqual(len(loader.build_sequences(loader.dates[2], include_target=False)[0]), 2)
        with self.assertRaises(ValueError):
            loader.build_sequences(loader.dates[2])

    def test_future_changes_do_not_change_past_sequences(self):
        loader = self.load()
        original = loader.build_sequences(loader.dates[2], return_dict=True)['X']
        self.df.loc[self.df.date >= loader.dates[2], 'pca_0'] = 999.
        loader = self.load()
        np.testing.assert_array_equal(original, loader.build_sequences(loader.dates[2], return_dict=True)['X'])

    def test_month_end_and_month_start_keys_merge(self):
        features = self.df.drop(columns='return')
        returns = self.df[['date', 'asset', 'return']].copy()
        returns['date'] = returns.date + pd.offsets.MonthEnd(0)
        returns_path = self.path.with_name('returns.csv')
        returns.to_csv(returns_path, index=False)
        loader = self.load(features, returns_path=returns_path)
        self.assertEqual(loader.df['return'].notna().sum(), len(features))

    def test_partial_label_coverage_is_reported(self):
        self.df.loc[4, 'return'] = np.nan
        loader = self.load()
        batch = loader.build_sequences(loader.dates[2], return_dict=True)
        self.assertEqual(batch['coverage']['complete_histories'], 2)
        self.assertEqual(batch['coverage']['included_assets'], 1)

    def test_external_return_duplicates_are_rejected(self):
        path = self.path.with_name('returns.csv')
        pd.concat([self.df, self.df.iloc[:1]]).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            self.load(self.df.drop(columns='return'), returns_path=path)
