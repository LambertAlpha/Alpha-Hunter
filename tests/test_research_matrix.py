"""Exercise the declared comparison and paired, month-level aggregation."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.generate_fixture import generate
from scripts.run_research_matrix import paired_summary, run_matrix
from src.config import Config


class ResearchMatrixTests(unittest.TestCase):
    def test_paired_effect_averages_seeds_before_month_bootstrap(self):
        rows = [dict(date=d, seed=s, variant=v, ic=value + s / 1000)
                for d in ['2023-01', '2023-02', '2023-03', '2023-04'] for s in [13, 42, 101]
                for v, value in [('gated', .1), ('ungated', .08), ('gated_prediction_only', .11)]]
        output = paired_summary(pd.DataFrame(rows), [13, 42, 101], draws=100)
        self.assertAlmostEqual(output[0]['paired_mean_ic_difference'], .02)
        self.assertAlmostEqual(output[1]['paired_mean_ic_difference'], -.01)
        self.assertEqual(output[0]['months'], 4)
        with self.assertRaises(ValueError):
            paired_summary(pd.DataFrame(rows[:-1]), [13, 42, 101])

    def test_actual_matrix_checks_coverage_and_writes_complete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config.load(generate(root / 'fixture', months=16, assets=8))
            config.training.prediction_start = '2011-03'
            config.training.prediction_end = '2011-04'
            config.evaluation.ranking_only = True
            config.random_forest.n_estimators = 2
            for name in ['mlp', 'transformer', 'tfa']:
                getattr(config, name).epochs = 1
            result = run_matrix(config, root / 'runs', [42])
            manifest = json.loads((result / 'matrix.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['total_fits'], 14)
            self.assertEqual(manifest['total_predictions'], 112)
            self.assertEqual(len(pd.read_csv(result / 'monthly_ic.csv')), 14)


if __name__ == '__main__':
    unittest.main()
