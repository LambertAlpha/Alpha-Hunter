# Alpha-Hunter

Temporal representation learning for cross-sectional return prediction.

An academic team project at The Chinese University of Hong Kong, Shenzhen (2025),
by **Boyi Lin, Linyi Qian, and Tingyu Yan**. We explored whether auxiliary objectives
could improve a Transformer-based predictor's representations and empirical stability.

[Research report and archive notes](final_paper/README_paper.md) ·
[Reproduction guide](docs/reproducibility.md) ·
[Evaluation audit](docs/evaluation-audit.md) ·
[2026 engineering study](docs/engineering-study-2026-09-14.md)

## Research question

How does adding reconstruction and structural regularization change a temporal model's
predictions? Using monthly equity data as the experimental setting, we compared simple
baselines with a Temporal Factor Autoencoder (TFA), varied the number of PCA features,
and studied the effect of auxiliary loss weights.

## Method

The pipeline consumes sequences of precomputed PCA features. A Transformer encoder feeds
latent-factor and prediction heads; a decoder reconstructs the input. TFA predicts quantile
classes of rank targets, while the baseline models use cross-sectional rank targets.

The training objective combines four terms:

```math
\mathcal{L} = \mathcal{L}_{\mathrm{prediction}} + \alpha\mathcal{L}_{\mathrm{reconstruction}} + \beta\mathcal{L}_{\mathrm{smoothness}} + \gamma\mathcal{L}_{\mathrm{decorrelation}}
```

Reconstruction encourages information preservation, smoothness penalizes changes in an
auxiliary factor-weight head, and the correlation penalty encourages decorrelated latent
factors. These are design motivations, not guarantees of economic interpretability or
statistical independence. A residual feature gate now connects factor weights to the
prediction and reconstruction paths; uniform weights recover the ungated representation.
The ungated option remains available for ablations. The decoder also receives full encoder
memory, so this is not a strict latent bottleneck or a replication of a conditional
asset-pricing autoencoder. See the [implementation audit](docs/evaluation-audit.md).

| Component | Implementation |
| --- | --- |
| Sequence construction and separate rank/raw-return targets | [data_loader.py](src/data_loader.py) |
| Transformer, Ridge, random forest, and MLP baselines | [models.py](src/models.py) |
| TFA architecture and auxiliary objectives | [models_tfa.py](src/models_tfa.py) |
| Rolling training, validation, and prediction | [trainer.py](src/trainer.py) |
| Rank correlation and portfolio evaluation | [evaluator.py](src/evaluator.py) |

## Experiments and evidence

The project explored 11- and 31-component PCA inputs, baseline comparisons, removal of
reconstruction or smoothness/decorrelation penalties, and changes to reconstruction weight.
The preserved materials show the experimental process, including configurations that did
not improve predictive metrics.

| Artifact | What it contains |
| --- | --- |
| [2025 course report](final_paper/paper.pdf) | Original research narrative and figures; read with the audit below. |
| [11-component summary](report/11pca/summary.md) | Baselines and partial-run ablations. |
| [31-component summary](report/31pca/summary.md) | An alternative feature configuration. |
| [Archived results](report.zip) | Selected prediction CSVs, statistics, and figures. |

**Evaluation status, September 2026:** a reproducibility audit identified rank targets being
exported as economic returns, overlapping training/validation dates, and incorrect sequence
axis ordering in the published pipeline. The current code corrects these issues and includes
regression tests. A second pass corrects portfolio accounting, model checkpointing,
rolling error handling and configuration propagation, and tests gated/ungated TFA on a
controlled synthetic panel. These engineering checks are separate from market evidence.
**The historical Sharpe, drawdown, and “66% lower drawdown” claims are not
validated performance results.** Corrected full-data experiments have not been run, and some
historical comparisons also use different evaluation horizons. Details and evidence are in
[the audit](docs/evaluation-audit.md).

## My contribution

**Boyi Lin:** project lead; model development and experimental comparisons for the TFA
approach, including auxiliary-loss design and ablation analysis. This is a joint course
project with Linyi Qian and Tingyu Yan, not a sole-author publication. The report retains
all three authors. The September 2026 maintenance work adds an explicit evidence audit,
corrected data/evaluation plumbing, and regression coverage.

## Run and inspect

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m scripts.generate_fixture
uv run python train.py --model all --config results/synthetic-fixture/config.json
uv run python train_tfa.py --help
```

The tests use generated data and require no market-data credentials. Original feature and
return datasets are not included, so passing the tests does not reproduce the research
results. The [reproduction guide](docs/reproducibility.md) describes the input schema,
training commands, outputs, and remaining limits.
