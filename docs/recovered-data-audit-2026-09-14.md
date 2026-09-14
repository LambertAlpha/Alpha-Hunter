# Recovered source data and a fixed-basis diagnostic

The recovery found an older local checkout with 90,846 cleaned feature rows, 203
features, 1,678 securities and 212 monthly dates (January 2007–August 2024). Its
82,678-row PCA file contains 11 components over December 2008–August 2024. These
facts conflict with the paper's 2013–2023 / 132-month narrative. The later public
notebook instead requests 30 components and caches a 31-component output. The
local 11-component file must not be described as the public 31-component file.

The old component-selection code fits a capped PCA, searches that truncated
variance curve for 80%, and adds one. When the threshold is unreachable within
the cap, this produces cap+1 components. The recovered 11-component variance
ratios average 41.8475%, ranging from 40.9717% to 42.7401%. Neither a cap of 10
nor an 80% variance claim describes that output accurately.

Other limitations in the preserved notebook:

- Industry exposures are the first three ticker characters, not independently
  sourced industries. The purported size exposure is an accounting-to-market
  ratio. Neither should be advertised as conventional industry/size neutralization.
- A new PCA is fitted on every trailing 24-month window. Signs, order and rotations
  are not aligned before monthly scores are concatenated into neural sequences.
  A component index therefore does not guarantee a persistent feature meaning.
- Including a feature month's cross-section in its PCA fit is not automatically
  future leakage when the target is next month. The unresolved issue is whether
  upstream feature data were actually available by that feature month end.
- The final loading plot refits an additional standardized PCA, so it does not
  reconstruct the exact saved transform.

## Return source and timing

A separate DDA3600 coursework return panel was recovered. Two local copies have
identical SHA-256 hashes. It is a 291-by-5,124 monthly panel from January 2000 to
March 2024. The coursework code shifts this panel backward one row to pair
month-t characteristics with t+1 returns. This supports interpreting the stored
index as the month in which the return was earned; it is not a vendor audit.
Conversion to CSV retains missing values and neither shifts nor rescales labels.

Across the 206 overlapping next-month targets (February 2007–March 2024), every
stock in the previous feature-month cross-section has an observed candidate
return. We did not select the universe based on future label availability. This
coverage does not establish corporate-action correctness, delisting treatment,
or equivalence to the original paper's unlocated return input. Returns after
March 2024 are absent and have not been synthesized. Large returns in the broad
panel were not clipped merely because they were unusual.

The original acquisition/cleaning code, fundamental release timestamps, revision
policy, and independently verified historical constituent records have not been
recovered. A historical index label alone does not establish a point-in-time
universe. The existing feature-membership filter is an explicit proxy; trading at
a month-end price using inputs only finalized at that close is not validated.

## Corrected preparation

The new CSV-only preparation fits full-SVD PCA once on 8,516 calibration rows
from January 2007–December 2008. It exports only later feature months. Centering,
loadings and component selection are learned exclusively in this calibration
period. The hard cap is 10; these components explain 43.0022%, while 55 would be
needed to reach 80%. The unmet target is recorded rather than overridden.

The fixed basis eliminates moving component definitions in this experiment. Its
tradeoff is staleness: a 2007–2008 basis may lose relevance later. No superiority
of this calibration choice is claimed. The recovered cleaned features are used
as supplied, without additional winsorization, pseudo-industry neutralization or
cross-sectional scaling. Upstream point-in-time uncertainty remains.

The exported panel has 77,878 rows, 1,531 distinct securities and 179 monthly dates
(January 2009–November 2023). It retains separate next-month labels through
December 2023. Every exported member has a next-month label. Requiring a full
12-month feature history reduces the actual 2023 prediction cross-sections to
357–424 securities. Each model and seed must use exactly the same dates, assets
and raw returns. This conditional sample is not the entire CSI500 index.

## Interpretation of the matched experiment

The predeclared 2023 comparison is retrospective: the year was already part of
the course report. Training rolls forward monthly using 36 training and six
validation label months, so earlier 2023 outcomes can enter later training or
validation windows after they occur. There is no genuinely untouched holdout
claim. Model capacity and objectives differ across baseline families: TFA has 43,036
parameters, Transformer 4,033, and MLP 4,417 in this configuration. A shared
calendar and inputs are not a claim that all models have identical capacity.

Gated versus ungated TFA changes gate connectivity. Gated versus prediction-only
changes the auxiliary objectives while keeping validation selection on the same
prediction cross-entropy criterion. It tests the auxiliary objectives jointly,
not the separate contribution of each individual loss term. TFA uses classification over ranked
returns, while the baseline Transformer uses rank regression. A TFA–Transformer
difference therefore cannot by itself prove a benefit from latent factors.

No portfolio/Sharpe statistics are generated. Monthly IC, per-seed results and
paired month-level differences are descriptive. Random seeds are averaged within
months before a circular three-month block bootstrap; 12 months still provide
little information about persistence across market conditions.

## What would justify stronger claims

Recover the acquisition/cleaning source, release-date handling, historical
constituent records and return-vendor adjustment definitions first. Then lock a
new evaluation protocol before further model selection. A wider range of years,
capacity/objective controls and dimension-selection rules chosen only on
training/validation data would address different open questions; this short
retrospective matrix cannot replace them. More PCA variance is not automatically
more predictive information, so an 80% target should not itself become a model
selection criterion on evaluation outcomes.

The 2025 paper is preserved as a historical artifact. This audit and its new
results live in maintenance documentation and a separate report directory.

[Protocol, results and verification](../report/recovered_2026_09_14/README.md).
