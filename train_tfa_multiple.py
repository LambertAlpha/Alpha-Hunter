"""Predeclared TFA ablations on identical dates/seeds, each in its own directory.

These are diagnostics, not a license to choose a model on reported test returns.
Use a separate untouched holdout for any subsequent model-selection claim.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.cli import write_json

ABLATIONS = {
    'ungated': ['--no-factor-gating'],
    'gated': ['--factor-gating'],
    'gated_prediction_only': ['--factor-gating', '--alpha', '0', '--beta', '0', '--gamma', '0'],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', default='results/tfa-sweeps')
    parser.add_argument('--seeds', nargs='+', type=int, default=[13, 42, 101])
    args = parser.parse_args(argv)
    root = Path(args.output_dir) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in args.seeds:
        for name, flags in ABLATIONS.items():
            run_name = f'{name}-seed{seed}'
            command = [sys.executable, str(Path(__file__).with_name('train_tfa.py')),
                       '--config', str(Path(args.config).resolve()), '--seed', str(seed),
                       '--output-dir', str(root.resolve()), '--run-name', run_name, *flags]
            subprocess.run(command, check=True)
            path = root / run_name / 'tfa'
            results.append(dict(ablation=name, seed=seed, path=str(path),
                                stats=json.loads((path / 'stats.json').read_text())))
            write_json(root / 'summary.json', results)
    print(f'All predeclared ablations completed: {root.resolve()}')
    return root


if __name__ == '__main__':
    main()
