# Historical coursework materials — 2025

Supporting material for the CUHK-Shenzhen DDA3600 course project. The
[original manuscript](../../final_paper/README.md) has its own directory and reading notes.
These files preserve the historical coursework process; they are not current benchmark
results. Read the [evaluation audit](../../docs/evaluation-audit.md) before interpreting
any old performance numbers.

| Material | Role |
| --- | --- |
| [Presentation](presentation.pdf) | TFA course presentation, December 8, 2025. |
| [Report requirements](report-guidelines.pdf) | Course instructions and grading requirements. |
| [Original plotting script](figure-scripts/create_figures.py) | Historical figures using hard-coded summary values and workstation-specific output paths. |
| [Updated plotting script](figure-scripts/create_figures_updated.py) | Historical figures using a different capital/cost convention and relative output paths. |
| [Drawdown chart](figures/maxdd_reduction.png) | Historical chart not referenced by the final LaTeX manuscript. |
| [Sharpe/drawdown chart](figures/sharpe_maxdd_comparison.png) | Historical chart not referenced by the final LaTeX manuscript. |

The plotting scripts contain literal metrics; they do not load predictions or recompute
portfolio results. They are retained as historical source, not as a reproduction command
for current experiments. Use the [current reproduction guide](../../docs/reproducibility.md)
and [auditable synthetic study](../../report/engineering_2026_09_14/README.md) for the
maintained pipeline. Original file contents were preserved when this archive was organized.
