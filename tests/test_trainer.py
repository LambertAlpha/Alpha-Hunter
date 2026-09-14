"""Rolling failures, coverage and cache behavior."""
import unittest
from unittest.mock import patch

import numpy as np
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


if __name__ == '__main__':
    unittest.main()
