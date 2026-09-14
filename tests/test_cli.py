"""Configuration precedence and isolated run artifacts."""
import json
import tempfile
import unittest
from pathlib import Path

from src.config import Config


class ConfigTests(unittest.TestCase):
    def test_unknown_fields_fail_instead_of_disappearing(self):
        with self.assertRaises(ValueError):
            Config.from_dict({'tfa': {'dropuot': .5}})
        with self.assertRaises(ValueError):
            Config.from_dict({'unknown': {}})

    def test_config_snapshot_does_not_mutate_source(self):
        config = Config()
        snapshot = config.to_dict()
        snapshot['tfa']['dropout'] = .99
        self.assertNotEqual(config.tfa.dropout, .99)

    def test_cli_preserves_config_and_applies_explicit_override(self):
        from src.cli import parse_config
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps({'data': {'pca_path': 'custom.csv', 'sequence_length': 4},
                                        'tfa': {'dropout': .3, 'weight_decay': .04},
                                        'training': {'seed': 9}}))
            _, config = parse_config(['--config', str(path)], default_model='tfa')
            self.assertEqual(config.data.pca_path, 'custom.csv')
            self.assertEqual(config.training.seed, 9)
            self.assertEqual(config.tfa.dropout, .3)
            _, config = parse_config(['--config', str(path), '--dropout', '.2', '--weight_decay', '.05',
                                      '--early_stopping_patience', '7'], default_model='tfa')
            self.assertEqual(config.tfa.dropout, .2)
            self.assertEqual(config.tfa.weight_decay, .05)
            self.assertEqual(config.tfa.early_stopping_patience, 7)
