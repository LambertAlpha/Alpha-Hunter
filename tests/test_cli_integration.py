"""Tiny end-to-end runs exercise actual CLI artifacts and process failures."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.generate_fixture import generate
from src.cli import main, run_experiment
from src.config import Config
from src.data_loader import SequenceDataLoader
from src.models_tfa import TFAPredictor


class CLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = generate(self.root / 'fixture', months=16, assets=8)
        self.config = Config.load(self.config_path)
        for name in ['mlp', 'transformer', 'tfa']:
            getattr(self.config, name).epochs = 1
        self.config.random_forest.n_estimators = 3
        self.config.save(self.config_path)

    def test_five_models_have_identical_coverage_and_provenance(self):
        root = main(['--model', 'all', '--config', str(self.config_path),
                     '--output-dir', str(self.root), '--run-name', 'all'])
        expected = None
        for name in ['ridge', 'random_forest', 'mlp', 'transformer', 'tfa']:
            path = root / name
            status = json.loads((path / 'run.json').read_text())
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['successful_months'], 2)
            self.assertIn('pca', status['data_sha256'])
            keys = pd.read_csv(path / 'predictions.csv')[['date', 'asset']]
            if expected is not None:
                pd.testing.assert_frame_equal(keys, expected)
            expected = keys
        self.assertTrue((root / 'tfa' / 'last_model.pt').exists())

    def test_same_seed_repeats_scores_and_checkpoint_reloads(self):
        self.config.training.max_prediction_dates = 1
        for name in ['first', 'repeat']:
            run_experiment('tfa', self.config, self.root / name)
        first = pd.read_csv(self.root / 'first' / 'predictions.csv')
        repeat = pd.read_csv(self.root / 'repeat' / 'predictions.csv')
        pd.testing.assert_frame_equal(first, repeat, check_exact=True)
        predictor = TFAPredictor.load(self.root / 'first' / 'last_model.pt')
        loader = SequenceDataLoader(self.config.data.pca_path, sequence_length=3, forward_fill_limit=0)
        X = loader.build_sequences(pd.Timestamp(first.date.iloc[0]), return_dict=True)['X']
        np.testing.assert_allclose(predictor.predict(X), first.prediction.to_numpy(), rtol=1e-6)
        self.assertEqual(predictor.feature_names, loader.feature_columns)
        # The exact same forecast must be possible without any target/future
        # feature rows or labels, using a restored checkpoint and raw history.
        history = pd.read_csv(self.config.data.pca_path)
        history = history[pd.to_datetime(history.date) < pd.Timestamp(first.date.iloc[0])].drop(columns='return')
        path = self.root / 'past_features_only.csv'
        history.to_csv(path, index=False)
        past = SequenceDataLoader(path, sequence_length=3, forward_fill_limit=0)
        batch = past.build_sequences(pd.Timestamp(first.date.iloc[0]), include_target=False, return_dict=True)
        np.testing.assert_allclose(predictor.predict(batch['X']), first.prediction.to_numpy(), rtol=1e-6)

    def test_sparse_dates_get_no_portfolio_statistics(self):
        self.config.training.prediction_step = 2
        result = run_experiment('ridge', self.config, self.root / 'sparse')
        self.assertIn('portfolio_unavailable', result)
        self.assertFalse((self.root / 'sparse' / 'portfolio.csv').exists())

    def test_ranking_only_period_keeps_ic_without_portfolio(self):
        self.config.evaluation.ranking_only = True
        self.config.training.prediction_start = '2011-04'
        self.config.training.prediction_end = '2011-04'
        path = self.root / 'ranking'
        result = run_experiment('ridge', self.config, path)
        self.assertEqual(result['prediction_months'], 1)
        self.assertEqual(result['portfolio_unavailable'], 'Ranking-only diagnostic requested')
        self.assertFalse((path / 'portfolio.csv').exists())
        self.assertTrue((path / 'ic.csv').exists())

    def test_missing_test_label_partial_run_has_no_portfolio(self):
        frame = pd.read_csv(self.config.data.pca_path)
        first_test = sorted(frame.date.unique())[-2]
        frame.loc[(frame.date == first_test) & (frame.asset == 'A000'), 'return'] = np.nan
        frame.to_csv(self.config.data.pca_path, index=False)
        self.config.training.allow_skips = True
        path = self.root / 'partial'
        summary = run_experiment('ridge', self.config, path)
        self.assertIn('portfolio_unavailable', summary)
        self.assertFalse((path / 'portfolio.csv').exists())
        self.assertEqual(json.loads((path / 'run.json').read_text())['status'], 'partial')
        splits = json.loads((path / 'split_status.json').read_text())
        self.assertEqual([s['status'] for s in splits], ['failed', 'success'])
        self.assertEqual(splits[0]['coverage']['missing_return_assets'], ['A000'])
        self.assertEqual(splits[1]['val_missing_returns'][first_test], ['A000'])

    def test_failed_run_is_recorded_and_exits_nonzero(self):
        result = subprocess.run([sys.executable, 'train.py', '--model', 'ridge', '--config', str(self.config_path),
                                 '--pca-path', str(self.root / 'absent.csv'), '--output-dir', str(self.root),
                                 '--run-name', 'broken'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        status = json.loads((self.root / 'broken' / 'ridge' / 'run.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertNotIn('Completed', result.stdout)

    def test_existing_run_cannot_be_overwritten(self):
        self.root.joinpath('existing').mkdir()
        with self.assertRaises(FileExistsError):
            run_experiment('ridge', self.config, self.root / 'existing')

    def test_interrupted_run_does_not_keep_running_status(self):
        path = self.root / 'interrupted'
        with patch('src.cli.RollingWindowTrainer.train_and_predict', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_experiment('ridge', self.config, path)
        status = json.loads((path / 'run.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertIn('KeyboardInterrupt', status['error'])
