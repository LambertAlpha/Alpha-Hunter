# Course report: Temporal Factor Autoencoder

Dynamic Factor Investing with Balanced Temporal Factor Autoencoder

Boyi Lin, Linyi Qian, Tingyu Yan · CUHK-Shenzhen · December 2025

[Original PDF](paper.pdf) · [LaTeX source](paper.tex) · [Current project overview](../README.md)

## Reading the report today

This is the original academic course report, not a peer-reviewed publication or a newly
validated benchmark. It documents the project question, auxiliary objectives, baseline
experiments, ablations, and the interpretation made at the time.

A September 2026 audit found input-layout, validation-separation, and return-unit issues in
the published pipeline. The current code fixes the tested defects; full-data experiments
have not been repeated. The report's Sharpe/drawdown claims, including the 66% headline,
should be read with the [evaluation audit](../docs/evaluation-audit.md), not cited as
validated performance results.

## Supporting artifacts

- [11-component experiment summary](../report/11pca/summary.md)
- [31-component experiment summary](../report/31pca/summary.md)
- [Selected raw prediction/statistics archive](../report.zip)
- [Current reproduction guide](../docs/reproducibility.md)

The report uses some settings and coverage descriptions that differ from checked-in
configurations. Partial-run ablations, old 30-bps reports, and capital-constrained 10-bps
summaries are not interchangeable. The audit records these distinctions.

The original PDF and LaTeX are preserved. This README supersedes the previous promotional
summary without silently rewriting the historical report.

## Directory contents

This directory contains the manuscript and its required figure:

| File | Purpose |
| --- | --- |
| [paper.pdf](paper.pdf) | Original 2025 course report. |
| [paper.tex](paper.tex) | LaTeX source for that report. |
| [figures/risk_return_tradeoff.png](figures/risk_return_tradeoff.png) | Figure referenced by the LaTeX source; historical, unvalidated performance values. |

The presentation, course requirements, historical plotting scripts and unused charts are
in the [2025 coursework archive](../archive/coursework-2025/README.md). The manuscript,
PDF and referenced figure retain their original contents. Current experimental evidence
is presented separately in the [2026 engineering study](../docs/engineering-study-2026-09-14.md).
