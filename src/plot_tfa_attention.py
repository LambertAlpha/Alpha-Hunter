"""Visualize a saved TFA checkpoint with its fitted preprocessing.

This is post-hoc model inspection, not an out-of-sample performance estimate.
Example:
  uv run python -m src.plot_tfa_attention --pca_path /path/to/panel.csv \
      --model_path results/demo/run/tfa --sample_dates 1
"""
import argparse
from pathlib import Path

import pandas as pd

from .data_loader import SequenceDataLoader
from .models_tfa import TFAPredictor
from .tfa_analysis import TFAAnalyzer


def load_model(model_path: Path) -> TFAPredictor:
    checkpoint = model_path / 'last_model.pt' if model_path.is_dir() else model_path
    return TFAPredictor.load(checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pca_path', '--pca-path', required=True)
    parser.add_argument('--model_path', '--model-path', type=Path, required=True)
    parser.add_argument('--sample_dates', '--sample-dates', type=int, default=1)
    parser.add_argument('--forward-fill-limit', type=int, default=None,
                        help='Defaults to the neighboring saved config; otherwise specify explicitly')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.sample_dates < 1:
        parser.error('sample_dates must be positive')
    model = load_model(args.model_path)
    model_dir = args.model_path if args.model_path.is_dir() else args.model_path.parent
    fill = args.forward_fill_limit
    if fill is None:
        from .config import Config
        config_path = model_dir / 'config.json'
        if not config_path.exists():
            parser.error('Supply --forward-fill-limit when the saved config is unavailable')
        fill = Config.load(config_path).data.forward_fill_limit
    loader = SequenceDataLoader(args.pca_path, sequence_length=model.seq_len, forward_fill_limit=fill)
    if model.feature_names is not None and loader.feature_columns != model.feature_names:
        raise ValueError('Feature names/order differ from the checkpoint')
    output = args.output_dir or model_dir / 'checkpoint-inspection'
    output.mkdir(parents=True, exist_ok=True)
    analyzer = TFAAnalyzer(model)
    for date in loader.dates[-args.sample_dates:]:
        batch = loader.build_sequences(date, include_target=False, return_dict=True)
        weights = analyzer.extract_factor_weights(batch['X'], pd.DatetimeIndex([date] * len(batch['X'])),
                                                  batch['assets'].tolist())
        tag = date.strftime('%Y-%m')
        weights.to_csv(output / f'{tag}-weights.csv', index=False)
        analyzer.plot_average_attention_pattern(weights, save_path=output / f'{tag}-weights.png')
    print(f'Checkpoint inspection saved to {output.resolve()}')


if __name__ == '__main__':
    main()
