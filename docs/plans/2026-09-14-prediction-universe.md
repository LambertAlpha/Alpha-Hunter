# Prediction Universe Integrity Implementation Plan

**Goal:** Prevent target-month feature presence and missing future labels from silently selecting a backtest universe.

**Architecture:** Build prediction eligibility from membership in the last input feature month and a complete preceding feature window. Keep the return panel separate so label-only records survive. Historical training can use available labels with recorded exclusions; an evaluated test month must have every eligible asset's return or fail visibly. Inference also supports the month immediately after the observed feature calendar.

**Tech Stack:** Existing Python, pandas, NumPy, unittest and uv environment; no new dependencies.

## Alternatives and decision

1. Keep current filtering and only warn: cheapest, but still permits a portfolio selected with future label availability.
2. Use past feature membership and reject incomplete test labels: selected bounded repair; no invented returns or external data dependency.
3. Add a full point-in-time security master and execution simulator: stronger eventual research foundation, but requires unavailable data and is not claimed by this change.

## Tasks

1. Add `tests/test_prediction_universe.py` for future-row deletion, stale-name reappearance, next-month inference, separate delisting labels and missing-label failure. Run `uv run python -m unittest discover -s tests -p test_prediction_universe.py -v` against the current implementation and record the failures.
2. Update `src/data_loader.py` to keep feature and label panels distinct, derive complete sequences and eligibility from preceding feature months, retain separate label-only rows, and report missing eligible labels by asset.
3. Update `src/trainer.py` to reject test-month label exclusions before fitting/exporting a scored portfolio; preserve failed split coverage and existing partial-run behavior.
4. Extend integration coverage for partial runs and verify all five models still produce identical predictions on the complete synthetic fixture. Exercise inference with a restored TFA checkpoint and no target-month rows.
5. Document the changed universe contract and limits in README, reproduction guide and evaluation audit. Run scoped Ruff, Pyright, full unittest suite and end-to-end controlled fixture checks before committing and publishing.

## Limits

Last-input-month membership is an explicit proxy, not a verified tradability/security-master claim. Historical PCA fitting, publication lags, execution timing and delisting adjustments still require upstream validation. No market-performance gain or statistical significance is inferred.
