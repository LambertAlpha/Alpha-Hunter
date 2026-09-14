"""Run matched baselines and TFA ablations; keep private predictions local.

The matrix is written before fitting and never selects a winner or changes its
settings using evaluation scores. Public summaries contain monthly aggregates.
"""
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from scripts.prepare_research_data import month
from src.cli import run_experiment, sha256_file, source_metadata, write_json
from src.config import Config


def paired_summary(monthly: pd.DataFrame, seeds: list[int], draws: int = 20000) -> list[dict]:
    """Average paired seeds per month, then circular block-bootstrap months.

    Seeds are not independent economic observations. With short retrospective
    samples these intervals are descriptive, not confirmatory significance tests.
    """
    pivot = monthly[monthly.variant.isin(['gated', 'ungated', 'gated_prediction_only'])].pivot(
        index=['date', 'seed'], columns='variant', values='ic')
    if pivot.isna().to_numpy().any() or not (pivot.groupby(level='date').size() == len(seeds)).all():
        raise ValueError('Paired comparison requires every variant/seed/month and finite IC')
    output = []
    for comparison, alternative in [('gate_effect', 'ungated'), ('auxiliary_effect', 'gated_prediction_only')]:
        differences = pivot['gated'] - pivot[alternative]
        monthly_difference = differences.groupby(level='date').mean().to_numpy()
        n = len(monthly_difference)
        block = min(3, n)
        rng = np.random.default_rng(2026)
        starts = rng.integers(0, n, size=(draws, (n + block - 1) // block))
        indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
        interval = np.quantile(monthly_difference[indices].mean(axis=1), [.025, .975])
        output.append(dict(comparison=comparison, contrast=f'gated minus {alternative}',
                           paired_mean_ic_difference=float(monthly_difference.mean()),
                           seed_differences={str(k): float(v) for k, v in differences.groupby(level='seed').mean().items()},
                           months=n, seeds=seeds, bootstrap_block_months=block, bootstrap_draws=draws,
                           descriptive_95pct_interval=interval.tolist(),
                           limitation='Retrospective short sample; overlapping rolling fits; no multiple-testing adjustment'))
    return output


def fit_experiment(job):
    """Process-local initialization avoids races in global NumPy/Torch RNGs."""
    experiment, config_dict, output = job
    effective = Config.from_dict(config_dict)
    name, model, seed = experiment['variant'], experiment['model'], experiment['seed']
    effective.training.seed = seed
    if model == 'tfa':
        effective.tfa.factor_gating = name != 'ungated'
        if name == 'gated_prediction_only':
            effective.tfa.alpha = effective.tfa.beta = effective.tfa.gamma = 0.
    run_name = f'{name}-seed{seed}'
    print(f'Starting {run_name}', flush=True)
    started = perf_counter()
    stats = run_experiment(model, effective, output / run_name)
    print(f'Finished {run_name}', flush=True)
    return stats, perf_counter() - started


def run_matrix(config: Config, output: Path, seeds: list[int], workers: int = 1):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Distinct seeds required')
    if not config.training.prediction_start or not config.training.prediction_end:
        raise ValueError('Declare prediction_start and prediction_end before running the matrix')
    if config.training.prediction_step != 1 or config.training.max_prediction_dates is not None:
        raise ValueError('Matrix requires the complete declared evaluation period')
    if config.training.allow_skips:
        raise ValueError('Matrix cannot skip failed months')
    if not config.evaluation.ranking_only:
        raise ValueError('This diagnostic matrix requires evaluation.ranking_only=true')
    if not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    output.mkdir(parents=True, exist_ok=False)
    experiments: list[dict[str, Any]] = [dict(variant=name, model=name, seed=config.training.seed)
                   for name in ['ridge', 'random_forest', 'mlp', 'transformer']]
    experiments += [dict(variant=variant, model='tfa', seed=seed) for seed in seeds
                    for variant in ['ungated', 'gated', 'gated_prediction_only']]
    manifest: dict[str, Any] = dict(status='running', experiments=experiments, base_config=config.to_dict(),
                    workers=workers, **source_metadata(), matrix_script_sha256=sha256_file(Path(__file__)))
    write_json(output / 'matrix.json', manifest)
    expected_dates = pd.date_range(month(config.training.prediction_start), month(config.training.prediction_end), freq='MS')
    expected = None
    rows, summaries = [], []
    executor = None
    try:
        jobs = [(experiment, config.to_dict(), output) for experiment in experiments]
        if workers == 1:
            results = map(fit_experiment, jobs)
        else:
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
            results = executor.map(fit_experiment, jobs)
        for experiment, (stats, elapsed) in zip(experiments, results):
            name, seed = experiment['variant'], experiment['seed']
            run_name = f'{name}-seed{seed}'
            path = output / run_name
            predictions = pd.read_csv(path / 'predictions.csv', dtype={'asset': str})
            keys = predictions[['date', 'asset', 'actual_return']]
            if expected is not None and not keys.equals(expected):
                raise ValueError('Every model/seed must predict identical dates/assets and use identical returns')
            expected = keys
            if not pd.DatetimeIndex(pd.to_datetime(predictions.date.unique())).equals(expected_dates):
                raise ValueError('Predictions do not cover the entire declared calendar')
            splits = json.loads((path / 'split_status.json').read_text())
            if any(s['status'] != 'success' or s['train_missing_returns'] or s['val_missing_returns'] for s in splits):
                raise ValueError('Matrix requires full observed train/validation/test label coverage')
            ic = pd.read_csv(path / 'ic.csv', index_col=0).iloc[:, 0]
            if not np.isfinite(ic.to_numpy()).all():
                raise ValueError('Every evaluation month must have a defined IC')
            counts = predictions.groupby('date').size()
            rows.extend(dict(variant=name, seed=seed, date=date, ic=value, assets=int(counts.loc[date]))
                        for date, value in ic.items())
            summaries.append(dict(**experiment, **stats, seconds=elapsed,
                                  predictions_sha256=sha256_file(path / 'predictions.csv')))
            pd.DataFrame(rows).to_csv(output / 'monthly_ic.csv', index=False)
            write_json(output / 'summary.json', summaries)
            print(f'Finished {run_name}: {len(ic)} months', flush=True)
        write_json(output / 'paired_comparisons.json', paired_summary(pd.DataFrame(rows), seeds))
        manifest.update(status='complete', completed_runs=len(summaries),
                        total_fits=sum(s['prediction_months'] for s in summaries),
                        total_predictions=sum(s['prediction_rows'] for s in summaries))
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}', completed_runs=len(summaries))
        raise
    finally:
        write_json(output / 'matrix.json', manifest)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[13, 42, 101])
    parser.add_argument('--workers', type=int, default=1, help='Independent processes; RNG state is never shared')
    args = parser.parse_args()
    print(run_matrix(Config.load(args.config), args.output, args.seeds, args.workers))


if __name__ == '__main__':
    main()
