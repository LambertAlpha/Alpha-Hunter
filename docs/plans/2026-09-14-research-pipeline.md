# Research Pipeline Reliability and TFA Improvement Plan

**Goal:** make the monthly prediction pipeline reproducible, fail visibly on invalid experiments, and connect the advertised factor weights to predictions before running controlled comparisons.

**Architecture:** retain the existing NumPy/Pandas/sklearn/PyTorch implementation and public predictor APIs. Share the CLI execution path, validate the monthly panel at ingestion, make portfolio accounting explicit, and add a backward-comparable residual factor-gating switch. Keep archived research outputs unchanged and clearly separate synthetic engineering validation from financial evidence.

**Tech stack:** Python, uv, unittest, NumPy, Pandas, sklearn, PyTorch; Ruff and Pyright for touched code; CPU-controlled synthetic experiments and GitHub Actions.

## Decision and scope

The user explicitly delegated design and implementation decisions. Execute locally without another design/permission round. Existing repository clone is isolated from the user's normal projects and is clean at `6a4e073`.

Considered: (1) tune a larger/newer architecture; (2) repair validity, reproducibility and the disconnected factor head; (3) rewrite around another research framework. Choose (2): neither tuning nor a rewrite repairs misleading accounting or discarded experiment parameters. Architecture changes remain testable against the legacy ungated graph, with no promise of higher market returns.

Original PCA/return data is absent from the checkout, and no Alpha-Hunter directory was found in the known local project directory. Do not replace it with invented market data or treat synthetic results as empirical alpha. PCA fit timing, universe construction and delisting coverage remain upstream-data requirements.

## 1. Monthly data contract

Files: `src/data_loader.py`, `tests/test_data_integrity.py`.

Reproduce duplicate-key acceptance, invalid/infinite values, zero fill-limit failure, missing targets and calendar gaps. Validate identifiers, numeric PCA fields and month keys; reject ambiguous duplicate merges and missing whole months. Keep missing feature handling forward-only and distinguish inference without labels from supervised construction. Test that future feature changes do not alter past input windows.

## 2. Portfolio accounting and statistics

Files: `src/evaluator.py`, `tests/test_evaluator.py`.

Test by hand: opening a portfolio pays opening notional once; unchanged names can require rebalancing after price drift; short positions use signed weights; long/short baskets do not overlap; the initial NAV participates in maximum drawdown. Reject unsupported value weighting, invalid observations and silent return floors. Define turnover as half L1 traded risky weights, publish costs separately, fail on insolvency, and include the number of valid IC observations. Plotting must finish without a blocking interactive window.

## 3. TFA correctness and inference artifacts

Files: `src/models_tfa.py`, `src/models.py`, `src/nn_utils.py`, `src/tfa_analysis.py`, `tests/test_models.py`.

First reproduce singleton-batch covariance NaNs and failure to restore the best validation epoch at normal completion. Stabilize covariance/correlation and check finite losses. Ensure fit resets fit-dependent state. Use a residual factor gate: add the projected deviation of feature-gated input from uniform weights to encoded representations; uniform gating exactly recovers the old representation. Include an ungated switch for ablation. Verify prediction-loss gradients reach the weight head and interventions on weights affect predictions.

Save an inference checkpoint containing architecture, weights, fitted scaler and class boundaries. Reload with the safe tensor/primitive checkpoint path and verify prediction round-trips. Apply training normalization in interpretation helpers as well.

## 4. Rolling experiment integrity and bounded caching

Files: `src/trainer.py`, `tests/test_trainer.py`.

Validate split arguments; support zero validation when appropriate. Replace whole-window dataset caching with bounded per-date caching to reuse overlapping data without accumulating all rolling windows. Report every planned/successful/failed date, default to propagating failures, validate finite prediction shape, and retain the last successful model for requested analysis. A repeated fit should not silently contaminate another run's state.

## 5. One effective configuration and reproducible CLI

Files: `src/config.py`, `train.py`, `train_tfa.py`, `train_tfa_multiple.py`, supporting run utilities, `tests/test_cli.py`.

Use config-file values unless explicitly overridden; reject unknown configuration fields. Make both entry points accept the advertised options. Pass every sweep option, isolate output directories, use the current interpreter, and collect only the current sweep's results. Save an effective configuration plus seed, input hashes, code revision, package versions and split coverage. Never print success after a failed or empty experiment. Use identical dates and seeds for comparisons; don't select a model by repeatedly optimizing on its reported test period.

## 6. Validation, experiments and publication

Files: meaningful regression tests, small synthetic-data/experiment helpers, `.github/workflows/ci.yml`, `docs/reproducibility.md`, `docs/evaluation-audit.md`, `README.md`, and a new research/engineering findings report.

1. Existing five regression tests must continue passing.
2. Record new test failures against the current implementation before fixes.
3. Run all regression tests; focused lint and type checks must pass before commits.
4. Run all supported models on a small deterministic monthly panel; verify artifacts, date coverage and repeated-seed behavior.
5. Run controlled TFA gated/ungated and auxiliary-loss comparisons on synthetic data. Report all chosen configurations and seeds, including negative findings, with costs and sample coverage. Treat this as software/learning-behavior validation only.
6. Document any research question that still needs licensed/raw data, rather than attaching an unsupported performance claim.
7. Commit, publish authorized changes, and verify the remote revision and clean working tree.

## Primary references checked

- [Gu, Kelly and Xiu, Autoencoder Asset Pricing Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3335536): their conditional asset-pricing model is not the same as reconstructing PCA sequences; do not claim a faithful replication.
- [PyTorch saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): best-weight copies, model reconstruction and safe inference loading.
- [Cvxportfolio turnover definition](https://www.cvxportfolio.com/en/1.2.0/constraints.html): half L1 trade weights excluding cash.
- [Chronological validation](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html): preserve temporal order and make split assumptions explicit.

## Execution evidence

Completed all six stages in source commit `079e923`. The final study ran against this
clean committed source: five model runs plus nine predeclared TFA ablations, 84 rolling
fits, 1,680 predictions, no skipped months. Published metrics were independently recomputed
from the consolidated prediction CSV; source and evidence SHA-256 hashes were verified.

Validation: 43 tests pass in the new locked macOS environment and Linux GitHub Actions;
scoped Ruff and Pyright pass. End-to-end checks include all five models, repeat-seed exact
predictions, inference checkpoint roundtrip, config precedence, sparse-period handling,
nonzero failure exits, plotting, and independently loading/visualizing a saved checkpoint.

The gate does not show a three-seed average IC improvement on this fixture (ungated .8579,
gated .8562). Report all seeds and variants in `docs/engineering-study-2026-09-14.md`.
Original-market-data revalidation remains a data dependency; no financial improvement claim
is made. Archived 2025 reports are preserved with their audit notices.
