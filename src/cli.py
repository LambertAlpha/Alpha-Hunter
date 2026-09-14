"""Shared, explicit configuration and reproducible experiment entry points."""
import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data_loader import SequenceDataLoader
from .evaluator import PerformanceEvaluator
from .models import MLPPredictor, RandomForestPredictor, RidgePredictor, TransformerPredictor
from .models_tfa import TFAPredictor
from .trainer import RollingWindowTrainer

MODELS = ['ridge', 'random_forest', 'mlp', 'transformer', 'tfa']


def write_json(path: Path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + '\n')


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def parse_config(argv=None, default_model='transformer'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--model', choices=MODELS + ['all', 'rf'], default=default_model)
    parser.add_argument('--run-name', help='Unique output subdirectory; existing runs are never overwritten')
    bindings = {}
    def option(name, section, key, converter=str, boolean=False):
        aliases = list(dict.fromkeys(['--' + name, '--' + name.replace('_', '-')]))
        bindings[name] = (section, key)
        if boolean:
            parser.add_argument(*aliases, action=argparse.BooleanOptionalAction, default=None, dest=name)
        else:
            parser.add_argument(*aliases, type=converter, default=None, dest=name)
    for name, converter in [('pca_path', str), ('returns_path', str), ('sequence_length', int), ('forward_fill_limit', int)]:
        option(name, 'data', name, converter)
    for name, converter in [('output_dir', str), ('seed', int), ('threads', int), ('train_window', int), ('val_window', int),
                            ('min_train_months', int), ('max_prediction_dates', int), ('prediction_step', int),
                            ('prediction_start', str), ('prediction_end', str)]:
        option(name, 'training', name, converter)
    for name in ['verbose', 'save_models', 'allow_skips', 'plot', 'analyze']:
        option(name, 'training', name, boolean=True)
    for name, converter in [('d_model', int), ('n_heads', int), ('n_encoder_layers', int), ('n_decoder_layers', int),
                            ('n_latent_factors', int), ('n_classes', int), ('alpha', float), ('beta', float), ('gamma', float)]:
        option(name, 'tfa', name, converter)
    option('factor_gating', 'tfa', 'factor_gating', boolean=True)
    option('ranking_only', 'evaluation', 'ranking_only', boolean=True)
    for name, converter in [('lr', float), ('weight_decay', float), ('batch_size', int), ('epochs', int),
                            ('early_stopping_patience', int), ('dropout', float), ('device', str)]:
        option(name, 'selected_neural', name, converter)
    args = parser.parse_args(argv)
    config = Config.load(args.config) if args.config else Config()
    args.model = 'random_forest' if args.model == 'rf' else args.model
    for name, (section, key) in bindings.items():
        value = getattr(args, name)
        if value is None:
            continue
        if section == 'selected_neural':
            targets = ['mlp', 'transformer', 'tfa'] if args.model == 'all' else [args.model]
            for target in targets:
                if target not in ['mlp', 'transformer', 'tfa'] or not hasattr(getattr(config, target), key):
                    if args.model != 'all':
                        parser.error(f'--{name} does not apply to {target}; use the corresponding JSON section')
                    continue
                setattr(getattr(config, target), key, value)
        else:
            if section == 'tfa' and args.model not in ['tfa', 'all']:
                parser.error(f'--{name} applies to TFA; use JSON for other model-specific settings')
            setattr(getattr(config, section), key, value)
    return args, config


def get_model_factory(model_name: str, config: Config, device=None, data_loader=None):
    if data_loader is None:
        raise ValueError('A data loader is required to establish actual feature dimensions')
    name = 'random_forest' if model_name == 'rf' else model_name
    constructors = dict(ridge=RidgePredictor, random_forest=RandomForestPredictor, mlp=MLPPredictor,
                        transformer=TransformerPredictor, tfa=TFAPredictor)
    if name not in constructors:
        raise ValueError(f'Unknown model: {name}')
    params = asdict(getattr(config, name))
    if name in ['mlp', 'transformer']:
        params['input_dim'] = data_loader.n_features * (data_loader.sequence_length if name == 'mlp' else 1)
    if device is not None and name in ['mlp', 'transformer', 'tfa']:
        params['device'] = device
    def factory(rolling_index: int = 0):
        rolling_seed = config.training.seed + rolling_index
        seed_everything(rolling_seed)
        local = dict(params)
        if name == 'random_forest':
            local['random_state'] = rolling_seed
        model = constructors[name](**local)
        if isinstance(model, TFAPredictor):
            model.feature_names = list(data_loader.feature_columns)
        return model
    return factory


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_metadata():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    source_files = sorted(list((root / 'src').glob('*.py')) + list(root.glob('train*.py')))
    return dict(git_revision=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
                source_sha256={str(p.relative_to(root)): sha256_file(p) for p in source_files},
                python=platform.python_version(), platform=platform.platform(),
                packages={p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch']})


def run_experiment(model_name: str, config: Config, output_dir: Path):
    """A single run writes an effective config, predictions, diagnostics and status."""
    config = Config.from_dict(config.to_dict())
    if output_dir.exists():
        raise FileExistsError(f'Run directory already exists: {output_dir}')
    output_dir.mkdir(parents=True)
    metadata: dict[str, Any] = dict(status='running', model=model_name, command=sys.argv, **source_metadata())
    write_json(output_dir / 'run.json', metadata)
    try:
        if config.training.threads < 1:
            raise ValueError('threads must be positive')
        torch.set_num_threads(config.training.threads)
        config.data.pca_path = str(Path(config.data.pca_path).resolve())
        if config.data.returns_path:
            config.data.returns_path = str(Path(config.data.returns_path).resolve())
        loader = SequenceDataLoader(**asdict(config.data))
        config.tfa.n_pca_factors, config.tfa.seq_len = loader.n_features, loader.sequence_length
        config.training.output_dir = str(output_dir.resolve())
        model_config = getattr(config, model_name)
        if hasattr(model_config, 'device') and model_config.device == 'auto':
            model_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_name == 'random_forest':
            config.random_forest.random_state = config.training.seed
        write_json(output_dir / 'config.json', config.to_dict())
        metadata.update(seed=config.training.seed, rolling_seed_rule='base seed + eligible calendar offset / prediction_step',
                        data_sha256={name: sha256_file(Path(path)) for name, path in
                                     [('pca', config.data.pca_path), ('returns', config.data.returns_path)] if path},
                        feature_names=loader.feature_columns, sequence_length=loader.sequence_length,
                        prediction_universe='last input feature-month membership with complete preceding history',
                        test_label_policy='require returns for every eligible asset; missing labels fail the month',
                        dataset_statistics={k: str(v) if k == 'date_range' else v for k, v in loader.get_statistics().items()})
        write_json(output_dir / 'run.json', metadata)
        trainer = RollingWindowTrainer(loader, get_model_factory(model_name, config, data_loader=loader),
                                       train_window=config.training.train_window, val_window=config.training.val_window,
                                       min_train_months=config.training.min_train_months, output_dir=output_dir)
        predictions = trainer.train_and_predict(save_models=config.training.save_models, verbose=config.training.verbose,
                         max_prediction_dates=config.training.max_prediction_dates, prediction_step=config.training.prediction_step,
                         prediction_start=config.training.prediction_start, prediction_end=config.training.prediction_end,
                         allow_skips=config.training.allow_skips,
                         save_last_model_path=output_dir / 'last_model.pt' if model_name == 'tfa' else None)
        evaluator = PerformanceEvaluator()
        ic = evaluator.compute_ic(predictions)
        ic.to_csv(output_dir / 'ic.csv')
        failed = any(s['status'] == 'failed' for s in trainer.run_status)
        if config.evaluation.ranking_only or config.training.prediction_step != 1 or failed:
            summary = dict(IC_mean=ic.mean(), IC_std=ic.std(), IC_valid_months=int(ic.notna().sum()),
                           prediction_months=predictions.date.nunique(), prediction_rows=len(predictions),
                           portfolio_unavailable=('Ranking-only diagnostic requested' if config.evaluation.ranking_only
                                                  else 'Sparse or failed months; no continuous monthly portfolio claimed'))
        else:
            evaluation = asdict(config.evaluation)
            evaluation.pop('ranking_only')
            annual = {k: evaluation.pop(k) for k in ['periods_per_year', 'risk_free_rate']}
            portfolio = evaluator.compute_portfolio_returns(predictions, **evaluation)
            portfolio.to_csv(output_dir / 'portfolio.csv')
            summary = evaluator.generate_summary_statistics(predictions, portfolio, **annual)
            if config.training.plot:
                evaluator.plot_performance(ic, portfolio, str(output_dir / 'performance.png'))
        write_json(output_dir / 'stats.json', summary)
        if config.training.save_models:
            trainer.save_models()
        assert trainer.last_model is not None and trainer.last_date is not None
        if hasattr(trainer.last_model, 'training_history'):
            write_json(output_dir / 'last_training_history.json', trainer.last_model.training_history)
        if config.training.analyze:
            if model_name != 'tfa':
                raise ValueError('Analysis requires model=tfa')
            from .tfa_analysis import TFAAnalyzer
            batch = loader.build_sequences(trainer.last_date, return_dict=True)
            TFAAnalyzer(trainer.last_model).generate_report(batch['X'], batch['raw_returns'],
                pd.DatetimeIndex([trainer.last_date] * len(batch['X'])), batch['assets'].tolist(), str(output_dir / 'analysis'))
        metadata.update(status='partial' if failed else 'complete',
                        planned_months=len(trainer.run_status), successful_months=predictions.date.nunique(),
                        last_model_date=str(trainer.last_date.date()))
        write_json(output_dir / 'run.json', metadata)
        return summary
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output_dir / 'run.json', metadata)
        raise


def main(argv=None, default_model='transformer'):
    args, config = parse_config(argv, default_model)
    run_name = args.run_name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    if Path(run_name).name != run_name or run_name in {'.', '..'}:
        raise ValueError('run-name must be a single directory name')
    root = Path(config.training.output_dir) / run_name
    names = MODELS if args.model == 'all' else [args.model]
    results = {}
    expected_keys = None
    for name in names:
        path = root / name
        results[name] = run_experiment(name, config, path)
        keys = pd.read_csv(path / 'predictions.csv', dtype={'asset': str})[['date', 'asset']]
        if expected_keys is not None and not keys.equals(expected_keys):
            raise ValueError('Model comparison requires identical date/asset coverage')
        expected_keys = keys
    write_json(root / 'comparison.json', results)
    print(f'Completed {len(results)} experiment(s): {root.resolve()}')
    return root
