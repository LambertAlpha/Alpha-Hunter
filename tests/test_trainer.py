"""Rolling failures, coverage and cache behavior."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import test_data_integrity as fixtures
from test_data_integrity import RecordingPredictor

from src.trainer import RollingWindowTrainer


class TrainerTests(unittest.TestCase):
    setUp = fixtures.DataIntegrityTests.setUp
    loader = fixtures.DataIntegrityTests.loader

    def trainer(self, **kwargs):
        return RollingWindowTrainer(self.loader(), RecordingPredictor, train_window=2,
                                    val_window=kwargs.pop('val_window', 1), min_train_months=2,
                                    output_dir=self.root / 'results', **kwargs)

    def test_failed_fit_cannot_silently_skip_a_month(self):
        trainer = self.trainer()
        with patch.object(RecordingPredictor, 'fit', side_effect=RuntimeError('broken fit')):
            with self.assertRaises(RuntimeError):
                trainer.train_and_predict(verbose=False)
        self.assertEqual(trainer.run_status[0]['status'], 'failed')

    def test_nonfinite_predictions_fail(self):
        with patch.object(RecordingPredictor, 'predict', return_value=np.array([np.inf, 0.])):
            with self.assertRaises(ValueError):
                self.trainer().train_and_predict(verbose=False)

    def test_interrupt_is_recorded_and_never_treated_as_skippable(self):
        trainer = self.trainer()
        with patch.object(RecordingPredictor, 'fit', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                trainer.train_and_predict(verbose=False, allow_skips=True)
        self.assertEqual(trainer.run_status[0]['status'], 'failed')
        self.assertIn('KeyboardInterrupt', trainer.run_status[0]['error'])

    def test_zero_validation_and_last_model(self):
        trainer = self.trainer(val_window=0)
        self.assertFalse(trainer.train_and_predict(verbose=False, max_prediction_dates=1).empty)
        self.assertIsNotNone(trainer.last_model)

    def test_date_cache_reuses_overlapping_months(self):
        trainer = self.trainer()
        dates = trainer.data_loader.dates
        with patch.object(trainer.data_loader, 'build_sequences', wraps=trainer.data_loader.build_sequences) as build:
            trainer._build_dataset(dates[2:4])
            trainer._build_dataset(dates[3:5])
        self.assertEqual(build.call_count, 3)
        for date in dates[5:]:
            trainer._build_dataset([date])
        self.assertLessEqual(len(trainer._dataset_cache), trainer.cache_limit)

    def test_invalid_step_rejected(self):
        with self.assertRaises(ValueError):
            self.trainer().train_and_predict(prediction_step=-1, verbose=False)

    def test_selected_period_preserves_training_history_and_model_seed(self):
        calls = []
        trainer = self.trainer()
        def factory(rolling_index):
            calls.append(rolling_index)
            return RecordingPredictor()
        trainer.model_factory = factory
        complete = trainer.train_and_predict(verbose=False)
        original_calls = list(calls)
        dates = sorted(complete.date.unique())
        calls.clear()
        selected = trainer.train_and_predict(verbose=False, prediction_start=str(dates[-1]),
                                              prediction_end=str(dates[-1]))
        self.assertEqual(calls, [original_calls[-1]])
        pd.testing.assert_frame_equal(selected.reset_index(drop=True),
                                      complete[complete.date.eq(dates[-1])].reset_index(drop=True))
        status = trainer.run_status[0]
        self.assertLess(max(status['train_dates']), min(status['val_dates']))
        self.assertLess(max(status['val_dates']), status['date'])

    def test_unavailable_reversed_or_invalid_period_fails(self):
        trainer = self.trainer()
        dates = trainer.data_loader.dates
        for bounds in [dict(prediction_start='1999-01'), dict(prediction_end='NaT'),
                       dict(prediction_start=str(dates[-1]), prediction_end=str(dates[-2]))]:
            with self.assertRaises(ValueError):
                trainer.train_and_predict(verbose=False, **bounds)


if __name__ == '__main__':
    unittest.main()
