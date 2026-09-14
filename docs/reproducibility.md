# Reproducing Alpha-Hunter

The original market feature store is not distributed. The synthetic example below runs
without accounts or credentials and tests software behavior, **not investment performance**.
Historical outputs predate the repairs in the [evaluation audit](evaluation-audit.md).

## Environment and checks

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run ruff check
uv run pyright
```

`uv.lock` describes the new environment, not the unavailable 2025 environment. The legacy
`requirements.txt` also includes optional notebook tools; the locked CLI environment does
not require Jupyter. Runs record their actual Python/package versions, platform, source
hashes, Git revision, seed and input hashes. CPU with one thread is the reproducibility
reference; exact numerical equality across hardware/library versions is not promised.

## An executable example

```bash
uv run python -m scripts.generate_fixture
uv run python train.py --model all \
  --config results/synthetic-fixture/config.json \
  --output-dir results/demo --run-name five-models
uv run python train_tfa_multiple.py \
  --config results/synthetic-fixture/config.json \
  --output-dir results/ablations --seeds 13 42 101
```

The fixture has 20 monthly dates, 20 assets and three features. The preceding three feature
months predict each label month. Each roll uses eight training-label months, three disjoint
validation-label months, and one test month: six test months, 120 predictions per model.
The signal deliberately depends on the preceding month's features; this is a controlled
learning example. Do not compare its IC or annualized Sharpe with market results.

The sweep reports every predeclared variant and seed: ungated TFA, gated TFA, and gated TFA
with all auxiliary weights zero. It invokes the current Python interpreter, passes every
option, and isolates outputs. It does not nominate a winner based on test returns.

## Input contract

| File | Required columns | Meaning |
| --- | --- | --- |
| Feature CSV | `date`, `asset`, `pca_0`, `pca_1`, … | Point-in-time monthly feature values. |
| Optional separate returns CSV | `date`, `asset`, `return` | Signed simple returns in decimal units. |

Alternatively include `return` in the feature CSV. Supplying both sources is an error.
Asset identifiers are strings, preserving leading zeros. Dates are normalized to month
starts; two observations for the same asset/month are rejected. Entire missing calendar
months, nonnumeric PCA fields and infinite values are rejected. Asset returns below −100%
are rejected; −100% is permitted as a total loss. Upstream adjustments must include
corporate actions and delistings.

Missing features can be forward-filled **within a feature, forward in time only**, limited
to the input window. `forward_fill_limit: 0` disables filling. An asset needs a complete
input window and presence in the target month's supplied panel. Supervised construction
requires returns; absent asset-level labels are excluded and coverage is recorded. This
filtering can create selection bias: report universe/label exclusions and resolve missing
or delisted assets upstream. The loader cannot infer a tradable point-in-time universe.
Inference within the supplied calendar can use `include_target=False` without labels;
future-month inference beyond the panel is not implemented.

PCA components must be fitted without future observations and remain aligned across
months (including signs/rotations if PCA is refitted). Merely using preceding feature dates
does not certify point-in-time data. The historical PCA notebook is not a complete,
audited data preparation pipeline.

## Configuration and commands

Both entry points accept `--config`; absent CLI options preserve file values. Unknown JSON
sections/keys and unknown flags fail. Use underscore or hyphen flag spellings. Model-specific
baseline parameters belong in `ridge`, `random_forest`, `mlp` or `transformer` JSON sections;
TFA exposes architecture/loss flags as well. `--epochs` and other common neural flags apply
to the selected neural model (or supported neural models in `--model all`).

```bash
uv run python train.py --model ridge \
  --pca-path /path/to/pca.csv --returns-path /path/to/returns.csv
uv run python train_tfa.py --config /path/to/experiment.json \
  --device cpu --factor-gating --alpha 0.05 --epochs 50
uv run python train_tfa.py --config /path/to/experiment.json \
  --max-prediction-dates 1 --epochs 1 --analyze --plot
```

Default chronology: 36 input months, 60 training-label months, 12 validation-label months;
at least 109 input months are required. `val_window: 0` disables validation; otherwise the
best validation epoch is restored even when the epoch budget finishes normally. TFA uses
validation **prediction cross-entropy** for stopping/scheduling, making auxiliary-loss
ablations comparable on the same criterion. Refit resets weights, optimizer, history,
normalization and label bins. Each rolling model receives `base seed + attempted roll index`.

`data.sequence_length` and actual feature columns determine TFA input dimensions; the
saved effective configuration records those derived values. `--max-prediction-dates N`
limits the first N eligible calendar dates before `--prediction-step` is applied. A step
above one produces ranking diagnostics only; missing months cannot be compounded into a
continuous monthly portfolio. Errors stop the run by default. Explicit `--allow-skips`
records partial coverage and disables portfolio statistics if any month fails.

## Outputs and independent TFA inference

Runs write to `output_dir/<unique-run-name>/<model>/`. Existing model run directories are
never overwritten. Check `run.json`: `complete`, `partial`, or `failed`.

- `config.json`: effective settings, including actual feature dimensions.
- `run.json`: provenance, hashes, dimensions and completion status.
- `split_status.json`: every planned date, train/validation dates, sample counts, coverage and errors.
- `predictions.csv`, `ic.csv`, `stats.json`: raw economic returns, ranking scores and diagnostics.
- `portfolio.csv`: consecutive-month gross/net returns, half-L1 turnover, traded notional and costs.
- `last_model.pt` (TFA): architecture, weights, fitted scaler, label boundaries and feature order.
- `last_training_history.json`: last rolling model's losses; optional nonblocking figures and analysis.

Undefined metrics are JSON `null`, never nonstandard `NaN`. A failed/empty experiment exits
nonzero. The saved checkpoint is for inference, not optimizer-state training resumption.

```python
from src.models_tfa import TFAPredictor

predictor = TFAPredictor.load("results/demo/five-models/tfa/last_model.pt")
# X: raw PCA sequences in predictor.feature_names order, matching sequence length
scores = predictor.predict(X)
```

Scores are in training-label units: this pipeline trains on percentile ranks, so a score of
0.8 is **not** an 80% expected return. Analysis through `TFAAnalyzer(predictor)` applies the
same fitted scaling. A bare network requires pre-scaled inputs. To inspect a checkpoint independently:

```bash
uv run python -m src.plot_tfa_attention \
  --pca-path results/synthetic-fixture/synthetic_panel.csv \
  --model-path results/demo/five-models/tfa --sample-dates 1
```

The inspector verifies feature order and reads the saved fill policy. This is post-hoc
inspection of one fixed model, not another walk-forward evaluation. Factor weights indicate
model gating behavior, not causal importance or demonstrated trading signals.

## Portfolio convention and research limits

At each month start select disjoint top/bottom baskets with deterministic asset tie-breaking.
Default risky weights are +0.5 long and −0.5 short (gross exposure 1). Residual cash earns
zero. Rebalance from the previous **post-return** weights, normalized by net NAV. Cost is
`one_way_rate * sum(abs(weight_trades))`; initial entry is charged once. Turnover is half
that L1 notional, excluding cash. Cost is paid from cash relative to pretrade NAV. We omit
borrow fees, cash interest, impact, execution delays and terminal liquidation; these must
be added before interpreting an implementable strategy. Value weighting is unsupported.
Portfolio insolvency raises an error rather than clipping the loss. Drawdown includes
initial NAV = 1. Long-leg statistics are gross basket diagnostics, not a separate net strategy.

Matched dates, multiple seeds, untouched selection/test data, costs, data availability and
negative findings are required for a research claim. This project does not implement a
complete nested hyperparameter search or prove the validity of its upstream market data.
