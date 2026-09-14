"""Generate a deterministic synthetic monthly panel; no financial claims."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.cli import write_json


def generate(output: Path, months: int = 20, assets: int = 20, seed: int = 2026):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(months, assets, 3))
    for month in range(1, months):
        features[month] = .4 * features[month - 1] + .9 * features[month]
    rows = []
    for month, date in enumerate(pd.date_range('2010-01-01', periods=months, freq='MS')):
        previous = features[max(0, month - 1)]
        signal = 1.3 * previous[:, 0] - .4 * previous[:, 1] + .6 * previous[:, 0] * previous[:, 2]
        returns = .035 * np.tanh(signal) + .01 * rng.normal(size=assets)
        for asset in range(assets):
            rows.append(dict(date=date, asset=f'A{asset:03d}',
                             **{f'pca_{j}': features[month, asset, j] for j in range(3)},
                             **{'return': returns[asset]}))
    path = output / 'synthetic_panel.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    config = dict(data=dict(pca_path=str(path.resolve()), sequence_length=3, forward_fill_limit=0),
                  training=dict(train_window=8, val_window=3, min_train_months=8, seed=42,
                                verbose=False, threads=1),
                  tfa=dict(d_model=16, n_heads=2, n_encoder_layers=1, n_decoder_layers=1,
                           n_latent_factors=3, dropout=0., epochs=12, batch_size=32,
                           early_stopping_patience=5, lr=.002),
                  transformer=dict(d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
                                   dropout=0., epochs=12, batch_size=32, lr=.002),
                  mlp=dict(hidden_dims=[32, 16], dropout=0., epochs=12, batch_size=32),
                  random_forest=dict(n_estimators=30, max_depth=5, min_samples_split=5, n_jobs=1),
                  evaluation=dict(transaction_cost=.001, long_pct=.2, short_pct=.2))
    write_json(output / 'config.json', config)
    write_json(output / 'fixture.json', dict(kind='synthetic', seed=seed, months=months, assets=assets,
                                           purpose='Engineering and learning-behavior validation, not market evidence'))
    return output / 'config.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/synthetic-fixture'))
    arguments = parser.parse_args()
    print(generate(arguments.output))
