# 31 因子摘要（feature/pca_feature_store.csv，rank 目标）

数据与设置：序列 36，train/val 60/12，交易成本 30bps，returns 来自 `data/cleaned/monthly_returns_proxy.csv`。

核心指标（全量，2025-12-08）：
| 模型 | IC | ICIR | IC>0 | LS Sharpe | MaxDD | Turnover | 文件 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge | 0.0144 | 0.1351 | 56.8% | 0.3709 | 0.6870 | 0.8781 | `more_pca_results/ridge/stats_20251207_234935.json` |
| Random Forest | 0.0361 | 0.2854 | 62.96% | 0.5787 | 0.7528 | 0.7043 | `more_pca_results/random_forest/predictions_20251208_005818.csv` |
| Transformer | 0.0054 | 0.0633 | 51.85% | 0.1159 | 0.5043 | 0.8672 | `more_pca_results/transformer/stats_20251208_112617.json` |
| MLP | -0.0122 | -0.1274 | 40.74% | -0.2422 | 0.8171 | 0.8333 | `more_pca_results/mlp/stats_20251208_104705.json` |
| TFA 优化版 | 0.0141 | 0.1288 | 59.26% | 0.0728 | 0.7364 | 0.8730 | `more_pca_results/tfa/predictions_20251208_022440.csv` |

要点：
- 增加到 31 因子后，IC/ICIR 总体低于 11 因子版本，信噪比下降。
- Random Forest 在 IC/ICIR 上最好，Transformer/TFA 表现一般，MLP 最弱。
- long-only Sharpe 极高需复核，不建议作为交易结论。
