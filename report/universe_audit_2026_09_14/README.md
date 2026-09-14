# Prediction-universe verification — September 14, 2026

This is software regression evidence on generated data, not a market-performance study.
The [audit](../../docs/evaluation-audit.md#third-maintenance-pass-prediction-universe-integrity)
explains the repaired selection mechanism and remaining assumptions.

## Verified behavior

- The local locked-environment suite passes **52 tests**, plus scoped Ruff and Pyright.
  Nine new tests cover future membership/label deletion, separate delisting labels,
  next-month inference, rolling calendar boundaries and partial-run handling. The existing
  checkpoint integration case also verifies predictions from past feature rows alone.
  The runtime source also passed [Linux CI](https://github.com/LambertAlpha/Alpha-Hunter/actions/runs/34894709087).
- All five baseline configurations were rerun from clean source
  [`16dcd4a`](https://github.com/LambertAlpha/Alpha-Hunter/commit/16dcd4a27e1dcd78c15ff49068dff762224ccc99),
  with the same synthetic fixture and seed as the previous engineering study.
  **Every date, asset, score and return matches exactly** after reading the CSVs:
  six months × 20 assets × five models = 600 prediction rows, from 30 rolling fits.
- A restored TFA checkpoint generates 20 scores for September 2011 using features through
  August 2011. Every return label is removed, and no September feature row is supplied.
  These are rank-target scores, not predicted economic returns or trades.

## Evidence files

| File | Contents |
| --- | --- |
| [verification.json](verification.json) | Source/data hashes, package versions, per-model equality checks and prediction hashes. |
| [forecast.csv](forecast.csv) | The 20 synthetic next-month scores without target rows or labels. |
| [SHA256SUMS](SHA256SUMS) | SHA-256 hashes of the two machine-readable files. |
| [Previous predictions](../engineering_2026_09_14/predictions.csv) | Unchanged reference results; select `run_id == baseline_<model>`. |

The input hashes and runtime source hashes identify the exact checked implementation.
Run-directory hashes refer to regenerated per-model CSVs; the old consolidated CSV also
contains an extra `run_id` column and other ablations, so its file hash differs by design.
No three-seed ablation was rerun in this follow-up, and no new superiority claim is made.

## Reproduce

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m scripts.generate_fixture --output results/universe-audit-fixture
uv run python train.py --model all \
  --config results/universe-audit-fixture/config.json \
  --output-dir results/universe-audit --run-name verified
```

Use a new run name if that directory already exists. Compare the resulting files:

```python
import pandas as pd

reference = pd.read_csv("report/engineering_2026_09_14/predictions.csv", dtype={"asset": str})
for model in ["ridge", "random_forest", "mlp", "transformer", "tfa"]:
    actual = pd.read_csv(f"results/universe-audit/verified/{model}/predictions.csv", dtype={"asset": str})
    expected = reference[reference.run_id == "baseline_" + model].drop(columns="run_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
```

Exact cross-platform floating-point reproduction is not guaranteed. The recorded equality
check used the macOS/Python/package environment in `verification.json`. Linux CI exercises
the behavioral suite, including repeated-seed and checkpoint round trips, separately.

The [inference example](../../docs/reproducibility.md#outputs-and-independent-tfa-inference)
shows how to construct the next-month batch. The stricter backtest contract intentionally
rejects incomplete future labels instead of reporting a portfolio on surviving observations.
Last-input-month membership still needs independent tradability and publication-time checks.
