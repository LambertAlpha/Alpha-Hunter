"""Temporal invariance, cap enforcement and unaltered missing-label behavior."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_research_data import frozen_pca, label_coverage, prepare


class ResearchPreparationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(33)
        self.features = pd.DataFrame([
            dict(date=date, asset=f'A{i}', **dict(zip(['a', 'b', 'c'], rng.normal(size=3))))
            for date in pd.date_range('2007-01-01', periods=6, freq='MS') for i in range(20)])

    def fit(self, frame):
        return frozen_pca(frame, '2007-01', '2007-03', '2007-06', max_components=1, variance_target=.99)

    def test_cap_does_not_grow_when_variance_target_is_unattainable(self):
        scores, metadata, basis = self.fit(self.features)
        self.assertEqual(metadata['components'], 1)
        self.assertFalse(metadata['variance_target_met'])
        self.assertEqual(metadata['components_needed_for_target'], 3)
        self.assertEqual(basis['components'].shape, (1, 3))
        self.assertEqual(list(scores.columns), ['date', 'asset', 'pca_1'])
        self.assertEqual(scores.date.min(), pd.Timestamp('2007-04-01'))

    def test_future_values_cannot_change_calibration_or_earlier_scores(self):
        scores, _, basis = self.fit(self.features)
        changed = self.features.copy()
        changed.loc[changed.date.ge('2007-06'), ['a', 'b', 'c']] *= 1000
        other, _, new_basis = self.fit(changed)
        np.testing.assert_array_equal(basis['components'], new_basis['components'])
        np.testing.assert_array_equal(basis['mean'], new_basis['mean'])
        pd.testing.assert_frame_equal(scores[scores.date.lt('2007-06')], other[other.date.lt('2007-06')])

    def test_missing_calendar_or_nonfinite_features_fail(self):
        with self.assertRaisesRegex(ValueError, 'calendar'):
            self.fit(self.features[self.features.date.ne('2007-02-01')])
        bad = self.features.copy()
        bad.loc[0, 'a'] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.fit(bad)

    def test_missing_returns_are_counted_without_future_membership_filter(self):
        scores, _, _ = self.fit(self.features)
        returns = scores[['date', 'asset']].copy()
        returns['date'] += pd.offsets.MonthBegin(1)
        returns['return'] = .01
        returns.loc[0, 'return'] = np.nan
        coverage = label_coverage(scores, returns)
        self.assertEqual(coverage.missing_labels.sum(), 1)
        self.assertEqual(coverage.eligible_members.sum(), len(scores))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.features.to_csv(root / 'features.csv', index=False)
            returns.to_csv(root / 'returns.csv', index=False)
            prepare(root / 'features.csv', root / 'returns.csv', root / 'output',
                    '2007-01', '2007-03', '2007-06', max_components=1)
            saved = pd.read_csv(root / 'output/returns.csv')
            self.assertEqual(saved['return'].isna().sum(), 1)
            self.assertEqual(saved.date.min(), '2007-05-01')


if __name__ == '__main__':
    unittest.main()
