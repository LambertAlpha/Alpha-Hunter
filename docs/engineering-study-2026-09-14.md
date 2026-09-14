# Engineering study — September 14, 2026

**Scope:** correctness and reproducibility of a course-project research pipeline. No
original market dataset was available, so this study does not establish financial alpha
or rehabilitate the historical Sharpe/drawdown claims.

## What changed

The [evaluation audit](evaluation-audit.md) documents the root causes and fixes. The main
changes are monthly data validation; signed-weight rebalancing and explicit costs; initial
capital in drawdown; finite singleton-batch losses; best-epoch restoration; an active
feature-gating path; self-contained TFA inference checkpoints; consistent analysis scaling;
and one shared configuration/experiment runner with visible failures and bounded caching.

Five existing regression tests passed before this second pass. Added accounting/panel cases
initially produced 14 failures and one error; added model/rolling cases reproduced nine
failures and two errors. Those behaviors now pass. The maintained test suite additionally
runs all five models, compares date/asset coverage, checks exact repeated-seed predictions,
reloads checkpoints, verifies CLI precedence, and checks process failure semantics.
The locked environment also passes scoped Ruff and Pyright checks.

## Controlled experiment protocol

Generate the fixture using `python -m scripts.generate_fixture`. Its data RNG seed is 2026;
it has 20 months, 20 assets and three synthetic features. Each feature follows an AR-like
process, and returns depend on the previous month's first two features plus an interaction
with the third. The label includes independent noise. This intentionally learnable process
is not calibrated to a market.

All runs use three input months, eight training-label months, three disjoint validation-label
months and six test months (March–August 2011): 120 predictions per model. TFA uses model
width 16, two heads, one encoder and decoder layer, three latent dimensions, five classes,
no dropout, batch size 32, learning rate .002, at most 12 epochs and patience five.
The full auxiliary weights are alpha=.1, beta=.05 and gamma=.01. Stopping uses validation
prediction cross-entropy. CPU, one thread; rolling seeds are base seed + roll index.

The baseline comparison uses seed 42. The ablation protocol declares seeds 13, 42 and 101
and reports all three variants: ungated, gated, and gated with auxiliary weights zero.
The ungated graph still benefits from all correctness fixes. It does not reproduce the
buggy historical experiments. No configuration is retuned after seeing these test results.

Preliminary controlled runs found mean IC near .858 for ungated TFA and .856 for gated
TFA across the three seeds. Two seeds improved with gating and one degraded. The final
committed-source rerun and machine-readable evidence are recorded below when published.
**This does not demonstrate an average predictive advantage from the gate.** The gate
repairs the disconnected mechanism; keeping it as the default is a design choice consistent
with the described model, not a test-set performance selection. The ungated switch remains
an explicit ablation.

## Interpretation and next research questions

1. **Fixing implementation validity is the strongest improvement here.** A prettier README
   or a larger Transformer would not repair rank returns, hidden failed months or costs.
2. **The architecture needs a sharper research hypothesis.** Full encoded memory reaches
   the decoder, so reconstruction does not show that the small latent vector is sufficient.
   A future comparison could distinguish sequence reconstruction from a true bottleneck,
   holding parameter count and stopping criteria constant.
3. **Weights are not causal attribution.** Gradients and interventions establish participation
   in computation. They do not identify economic factors or establish predictive usefulness.
   The old concentration diagnostic no longer labels concentration a “momentum signal.”
4. **Simple baselines deserve equal status.** Ridge and random forest learn this constructed
   signal well. Any real-data claim should survive these baselines, matched coverage, seeds,
   transaction/borrow costs and an untouched final holdout.
5. **Upstream data remains the main research dependency.** Audit PCA fit dates/component
   alignment, universe membership, delisting/label omissions and execution timing before
   interpreting a market backtest. Absent labels and asset selection can bias results even
   when the code uses strictly preceding feature dates.

No statistical significance is claimed from six test months or three seeds. Synthetic
annualized Sharpe can be very large by construction and is only an accounting diagnostic;
it is not a useful headline result for an application.
