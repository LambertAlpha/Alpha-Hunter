# Recovered-data diagnostic: locked protocol

This is a retrospective comparison using recovered course-project features and a separate
coursework return panel. It is not a reproduction of the 2025 paper, an untouched holdout,
or validated investment performance. The original cleaning source and vendor conventions
have not been recovered; see the forthcoming audit for the implications.

## Design declared before model evaluation

- Frozen PCA: fit all 203 cleaned features from January 2007 through December 2008 only;
  hard cap 10 components and report, rather than assume, the 80% variance target.
  Export January 2009–November 2023 scores in this single basis. No additional scaling,
  winsorization or pseudo-industry neutralization. The cleaned inputs are already normalized.
- Monthly targets: January–December 2023; 12 preceding feature months; 36 training-label
  months and six subsequent validation-label months. No forward-fill. All eligible
  test returns must be observed. Models may learn from earlier 2023 months as time advances.
- Four baselines at base seed 42: Ridge, random forest, MLP and Transformer. Three TFA
  variants (ungated, gated and gated prediction-only) at base seeds 13, 42, 101. This
  produces 13 runs and 156 independent monthly fits. Rolling seeds add the calendar offset
  to the base seed, identically across paired variants. Hyperparameters are in [protocol.json](protocol.json).
- Neural models use width 16, at most 10 epochs and batches of 512. Transformer/TFA
  early stopping uses validation loss and patience 3; MLP trains all 10 epochs.
  This is a modest-compute comparison, not equal-capacity models or an exhaustive search.
- Report all monthly IC values, all seeds and paired month-level differences. A circular
  three-month block bootstrap (20,000 draws, seed 2026) describes uncertainty after averaging
  paired seeds within each month. Twelve months do not establish generalization.
- No parameter/model selection on these evaluation scores. 2023 appeared in the historical
  project, so it cannot become an untouched holdout through this new protocol. No 2024
  model comparison is run here. Portfolio and Sharpe output is explicitly disabled.

Raw/processed data, security-level predictions and model checkpoints stay local while
redistribution rights remain unverified. Aggregate diagnostics and reproducible code may
be shared, with the data hashes and unresolved assumptions stated explicitly.
