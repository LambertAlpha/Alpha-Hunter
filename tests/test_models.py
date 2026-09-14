"""Model behavior and independently reloadable inference artifacts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.models_tfa import TemporalFactorAutoencoder, TFAPredictor
from src.nn_utils import PositionalEncoding

SMALL = dict(n_pca_factors=3, seq_len=2, d_model=8, n_heads=2,
             n_encoder_layers=1, n_decoder_layers=1, n_latent_factors=2,
             n_classes=2, dropout=0.)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        rng = np.random.default_rng(13)
        self.X = rng.normal(size=(5, 2, 3))
        self.y = np.array([.2, .4, .6, .8, 1.])

    def test_singleton_loss_and_gradients_are_finite(self):
        model = TemporalFactorAutoencoder(**SMALL)
        loss, _ = model.compute_loss(torch.tensor(self.X[:1], dtype=torch.float32), torch.tensor([0]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_prediction_loss_reaches_factor_weights(self):
        model = TemporalFactorAutoencoder(**SMALL)
        logits, _ = model(torch.tensor(self.X, dtype=torch.float32))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0, 1, 1, 1])).backward()
        gradients = [p.grad for p in model.factor_weight_generator.parameters()]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in gradients))

    def test_uniform_gate_matches_ungated_and_interventions_matter(self):
        model = TemporalFactorAutoencoder(**SMALL)
        X = torch.tensor(self.X, dtype=torch.float32)
        model.eval()
        model.factor_gating = False
        baseline = model(X)[0]
        model.factor_gating = True
        with patch.object(model.factor_weight_generator, 'forward', return_value=torch.ones_like(X) / 3):
            torch.testing.assert_close(model(X)[0], baseline)
        with patch.object(model.factor_weight_generator, 'forward', return_value=torch.tensor([1., 0., 0.]).expand_as(X)):
            self.assertFalse(torch.allclose(model(X)[0], baseline))

    def test_best_epoch_is_restored_at_normal_completion(self):
        predictor = TFAPredictor(**SMALL, epochs=3, batch_size=5, early_stopping_patience=10)
        original = predictor.model.compute_loss
        saved = []
        def controlled(*args, **kwargs):
            loss, components = original(*args, **kwargs)
            if not predictor.model.training:
                saved.append({k: v.clone() for k, v in predictor.model.state_dict().items()})
                components['prediction'] = float(len(saved))
                return loss * 0 + len(saved), components
            return loss, components
        with patch.object(predictor.model, 'compute_loss', side_effect=controlled):
            predictor.fit(self.X, self.y, self.X, self.y, verbose=False)
        for key, value in predictor.model.state_dict().items():
            torch.testing.assert_close(value, saved[0][key], rtol=0, atol=0)

    def test_inference_checkpoint_roundtrip_and_refit(self):
        predictor = TFAPredictor(**SMALL, epochs=2, batch_size=2)
        predictor.fit(self.X, self.y, verbose=False)
        expected = predictor.predict(self.X)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            predictor.save(path)
            restored = TFAPredictor.load(path)
            np.testing.assert_array_equal(restored.predict(self.X), expected)
        predictor.fit(self.X, self.y + 10, verbose=False)
        self.assertGreater(predictor.quantile_boundaries.min(), 10)
        self.assertEqual(len(predictor.training_history), 2)

    def test_odd_positional_dimension(self):
        result = PositionalEncoding(3)(torch.zeros(2, 2, 3))
        self.assertEqual(tuple(result.shape), (2, 2, 3))

    def test_transformer_restores_best_at_normal_completion(self):
        from src.models import TransformerPredictor
        predictor = TransformerPredictor(input_dim=3, d_model=8, nhead=2, num_layers=1,
                                         dim_feedforward=16, dropout=0, epochs=3,
                                         batch_size=5, early_stopping_patience=10)
        original = torch.nn.MSELoss.forward
        saved = []
        def controlled(loss_module, predicted, actual):
            loss = original(loss_module, predicted, actual)
            if not predictor.model.training:
                saved.append({k: v.clone() for k, v in predictor.model.state_dict().items()})
                return loss * 0 + len(saved)
            return loss
        with patch.object(torch.nn.MSELoss, 'forward', new=controlled):
            predictor.fit(self.X, self.y, self.X, self.y, verbose=False)
        for key, value in predictor.model.state_dict().items():
            torch.testing.assert_close(value, saved[0][key], rtol=0, atol=0)

    def test_analysis_uses_training_scaler(self):
        from src.tfa_analysis import TFAAnalyzer
        predictor = TFAPredictor(**SMALL, epochs=1, batch_size=5)
        predictor.fit(self.X * 100 + 50, self.y, verbose=False)
        raw = self.X * 100 + 50
        frame = TFAAnalyzer(predictor).extract_factor_weights(raw)
        direct = predictor.model.get_factor_weights(torch.tensor(predictor.transform_inputs(raw))).numpy()
        np.testing.assert_allclose(frame.weight.to_numpy(), direct.flatten())
        diagnostics = TFAAnalyzer(predictor).identify_attention_signals(frame)
        self.assertIn('high_concentration', diagnostics)
        self.assertNotIn('momentum_signal', diagnostics)
