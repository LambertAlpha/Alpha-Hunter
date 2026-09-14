# Synthetic study evidence

This is software/learning-behavior validation, not a market backtest. Read the
[study](../../docs/engineering-study-2026-09-14.md) and
[evaluation convention](../../docs/reproducibility.md) before interpreting numbers.

Source: `079e923c6a2165aee1dd7b50ad900720dfffbcc7`. All 14 runs used a clean checkout of this revision.

- [study.json](study.json): run status, environment, source/data hashes and all effective configurations.
- [split_status.json](split_status.json): complete per-run chronology and sample coverage.
- [predictions.csv](predictions.csv): all 1,680 prediction rows, with `run_id` added.
- [portfolios.csv](portfolios.csv): all 84 monthly portfolios, actual costs and traded weights' L1 norm.
- [metrics.csv](metrics.csv): all five models and all nine ablations; no model selected on these test results.
- [ablation_ic.png](ablation_ic.png): seed-by-seed ranking diagnostics.

Absolute host repository prefixes in JSON were replaced by `.`. This path substitution
does not change settings, input contents, scores or hashes. Generate the input using the
committed fixture generator; the original fixture was 20 months × 20 assets with seed 2026.
Model checkpoints are reproducible local outputs and are not duplicated in this evidence
folder. The CI/integration tests verify checkpoint inference roundtrips.

Synthetic Sharpe can be extremely large on this deliberately learnable process. It is
included in the full metric table for accounting transparency, not as a financial claim.
