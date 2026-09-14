# Reproducing and inspecting Alpha-Hunter

## What is available

The repository includes model/trainer source, regression tests, notebooks, the course
report, and selected archived predictions/statistics in [report.zip](../report.zip).
It does not include the original feature store, a complete set of predictions/checkpoints,
or a locked environment from the 2025 experiments.

The September 2026 code repairs are described in the [evaluation audit](evaluation-audit.md).
They change input layout, validation separation, and exported returns. New runs will not
reproduce the old reported metrics by construction.

## Environment

From the repository root:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run --no-project python -m unittest discover -s tests -v
uv run --no-project python train.py --help
uv run --no-project python train_tfa.py --help
```

The regression tests check feature/time ordering, feature-local forward filling,
rank/raw-return separation, chronological validation, and exported economic returns.
The older `python -m src.test_optimizations` script only checks imports and tracking when
market data is absent; it is not a full training or correctness test.

Dependency versions in requirements.txt are lower bounds, not a historical lockfile.
Record resolved versions, the Git revision, hardware, seed, and configuration for a new run.

## Required data

Supply data you have permission to use. No market-data provider credentials are needed
for the tests, and no dataset download is performed by the training commands.

| File | Required columns | Meaning |
| --- | --- | --- |
| PCA feature CSV | `date`, `asset`, `pca_0`, `pca_1`, … | Monthly, point-in-time feature values. |
| Returns CSV | `date`, `asset`, `return` | Realized signed simple returns in decimal units, e.g. `-0.02`. |

Use one row per asset/month, parseable dates, and matching asset/date keys across files.
The feature file may already include `return`; in that case its values take precedence
and the separate file is not merged. Do not supply ranks or percent units in `return`.
PCA fitting, universe membership, corporate actions, and source timing must be checked
upstream; the loader cannot certify the absence of look-ahead or survivorship bias.
The historical [PCA notebook](../pca.ipynb) is a reference, not a complete data release.

The default sequence is 36 months, followed by a rolling 60-month training-label window
and a disjoint 12-month validation-label window. The first eligible prediction is at
zero-based month index 108, so at least 109 distinct input months are required. Features
for each prediction end before its target month. Raw returns and cross-sectional ranks
are kept separately: ranks train the model; raw returns evaluate the portfolio.

## Training commands

The commands use the noninteractive Matplotlib backend so plotting does not wait for a
window to close. Replace the example paths with your own prepared files. These commands start new runs;
they do not restore the archived 2025 experiments.

```bash
MPLBACKEND=Agg uv run --no-project python train.py \
  --model ridge --device cpu \
  --pca_path /path/to/pca_features.csv \
  --returns_path /path/to/monthly_returns.csv \
  --output_dir results/ridge

MPLBACKEND=Agg uv run --no-project python train.py \
  --model transformer --device cpu \
  --pca_path /path/to/pca_features.csv \
  --returns_path /path/to/monthly_returns.csv \
  --output_dir results/transformer

MPLBACKEND=Agg uv run --no-project python train_tfa.py \
  --device cpu --epochs 50 \
  --alpha 0.02 --beta 0.01 --gamma 0.005 \
  --pca_path /path/to/pca_features.csv \
  --returns_path /path/to/monthly_returns.csv \
  --output_dir results/tfa_alpha002
```

`--max_prediction_dates 1` limits the eligible date range to its **first** prediction date;
it is useful for a smoke run, not a performance estimate. For TFA, `--epochs 1` reduces
training work further. Use the same prediction dates, source data, and evaluation settings
across comparisons. Select hyperparameters on development data and reserve untouched test
data; the repository does not implement an automated nested hyperparameter search.

Baseline settings can be supplied with `train.py --config config.json`. The TFA entry point
uses its explicit CLI options and does not accept `--config`, despite an old source comment.
The current evaluation default is 30 bps per side; the archived `final_report/*_10bps`
summaries use 10 bps per side and a different reporting convention. Do not combine them.

## Expected outputs and limitations

Runs write a configuration, timestamped prediction CSVs, portfolio CSVs, summary statistics,
and plots under `--output_dir`. Prediction rows contain `date`, `asset`, `prediction`, and
`actual_return`; the last column must contain raw economic returns. A TFA run may also
save the last model checkpoint. Check logs and row/date counts: a zero process exit code
alone does not guarantee successful training, because the legacy trainer catches failures.

Synthetic smoke tests validate execution only. There is no claimed expected market Sharpe,
IC, or drawdown for the corrected implementation. A complete study still requires a
permitted input dataset, matched horizons, multiple seeds, point-in-time feature checks,
and an explicit model-selection protocol.
