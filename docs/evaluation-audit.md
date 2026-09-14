# Evaluation audit — September 14, 2026

This note reviews the published project at commit `017882b` and documents maintenance
corrections. It distinguishes code/data integrity checks from a rerun of the research.
The original course report is retained as a historical artifact.

## Findings and corrections

| Finding | Evidence in the pre-audit code | Current handling |
| --- | --- | --- |
| Rank targets entered portfolio evaluation | `SequenceDataLoader` produced percentile-rank `y`; `RollingWindowTrainer` saved it as `actual_return`; the evaluator averaged that field as economic returns. | Preserve `raw_returns` separately and export those values. Regression tests use signed returns to catch a rank/return mix-up. |
| Validation labels overlapped training labels | `val_dates` was a suffix of `train_dates`. | Reserve a disjoint validation window immediately before the test month, with training before validation. |
| Time and feature axes were interchanged | The pivot produced `(feature, date)` columns but the reshape assumed `(date, feature)`. Forward filling could cross feature boundaries. | Fill within each feature, explicitly reindex to `(date, feature)`, then reshape. |

The regression suite failed on the old code and passes after these corrections. This
repairs the tested behaviors, not every possible modeling or backtesting issue.

## Consequences for historical results

The archived `report.zip` contains two 31-component prediction CSVs, each with 16,605 rows
across 81 dates from December 2017 through August 2024. Their `actual_return` values are
cross-sectional percentile ranks rather than signed financial returns. A portfolio return
series constructed from those values cannot support financial Sharpe or drawdown claims.
Spearman rank correlation does not have the same units problem, but training/validation
contamination and input-layout errors still prevent treating the old experiments as a
validated comparison of the intended models.

For the 11-component experiments, the archive preserves selected statistics but not the
complete prediction set needed to independently rebuild the final capital-constrained
comparison. The [experiment summary](../report/11pca/stats_summary.md) explicitly labels
random forest and ablations as partial 20-date runs. The main baselines and tuned runs
therefore cannot be assumed to share a test horizon.

The course report's **“66% lower maximum drawdown”** result and associated Sharpe values
are consequently **not endorsed as validated performance claims** by the current README.
The September 2026 maintenance did not retrain on the original market data or produce a
replacement headline metric.

## Method and documentation boundaries

- The report mentions a 12-month sequence; checked-in runtime configuration uses 36.
  Follow the saved run configuration when interpreting an experiment.
- The report describes nested walk-forward hyperparameter selection; the published code
  provides rolling training and early stopping, not a complete nested search pipeline.
- In `TemporalFactorAutoencoder.forward`, `weighted_pca` is computed but not consumed by
  the latent/prediction path. The auxiliary factor-weight head is regularized for
  smoothness; it should not be presented as a demonstrated causal feature attribution or
  as weights that directly determine predictions. The first maintenance pass did not change that architecture.
- A latent correlation penalty encourages decorrelation, not statistical independence.
- Upstream PCA timing and historical universe construction remain to be independently
  verified with the original data.
- The previous “10–50×” speed-up statement has no checked-in reproducible timing benchmark
  establishing its conditions. It is omitted from the current overview.

## Before making new performance claims

Use raw signed returns, point-in-time features, disjoint selection/test data, identical
evaluation dates and capital/cost conventions, and multiple seeds. Preserve predictions,
configurations, source versions, and a machine-readable table for every compared run.
Report negative results and feature-dimension sensitivity alongside the selected model.
See [reproducibility.md](reproducibility.md) for the current executable entry points.

## Second maintenance pass: model and experiment integrity

The subsequent autonomous review starts from `6a4e073`. New regression cases first
reproduced failures in portfolio accounting, monthly validation, model training and rolling
error handling. Current code additionally fixes the following:

| Finding | Correction and validation |
| --- | --- |
| Transaction costs depended on new names and doubled initial entry | Signed target weights, return drift, L1 traded notional and separate cost columns; hand-calculated tests. |
| First-month loss was omitted from drawdown; severe losses were floored | Initial NAV participates in running maximum; portfolio insolvency fails visibly. |
| Long/short baskets could overlap; “value” silently meant equal weighting | Require disjoint enabled legs and reject unsupported weighting. |
| Duplicates, calendar gaps, infinite inputs and zero fill limit were mishandled | Explicit monthly data contract, numeric checks, identifier preservation and tests. |
| Factor weights were disconnected from predictions | Residual feature gating now affects latent prediction and decoder memory; uniform weights recover the ungated graph; prediction-gradient/intervention tests. |
| A singleton batch caused covariance NaNs | Skip the correlation penalty when fewer than two samples are present; clamp variance and penalize off-diagonal correlations. |
| TFA and Transformer sometimes retained final rather than best validation weights | Restore best weights after normal completion and early stopping; reset fit-dependent state. |
| TFA checkpoints omitted preprocessing/architecture | Safe inference payload includes scaler, bins, dimensions and feature names; exact prediction roundtrip tested. |
| Analysis passed unscaled inputs to a trained network | Wrapped analysis uses the fitted predictor normalization. |
| Bad rolling months silently disappeared; caches grew with whole windows | Fail by default, record all planned dates/statuses, and bound caching by monthly slices. |
| CLI overwrote config-file values; sweeps dropped flags and mixed old outputs | Shared CLI, strict keys, effective settings and provenance, isolated runs, current-run-only comparisons. |

The older architecture-boundary bullet above describes `017882b`/the first maintenance
pass. It is superseded for current gated models. `factor_gating=False` remains available
for a controlled comparison of the ungated computation graph; it does not undo data,
accounting or training fixes.

The residual gate adds `W((F * weights - 1) * standardized_input)` to each encoded state,
where F is the number of features and W is the input projection without bias. Uniform
weights add zero. This is a testable design modification, **not an established improvement
on financial data**. The decoder still sees full sequence memory: the architecture is
not a strict latent bottleneck autoencoder, and its latent factors are not identified
asset-pricing factors. It is not a replication of
[Gu, Kelly and Xiu's conditional asset-pricing model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3335536).

Turnover follows the [half-L1 risky-trade convention](https://www.cvxportfolio.com/en/1.2.0/constraints.html).
Inference artifacts follow the [PyTorch model saving/loading guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
The new [engineering experiment report](engineering-study-2026-09-14.md) contains the
controlled checks and negative findings. Original-data research results remain unverified.
