# 11 因子摘要（data/pca/old/pca_feature_store.csv，rank 目标）

数据与设置：序列 36，train/val 60/12，交易成本 30bps，returns 来自 `data/cleaned/monthly_returns_proxy.csv`。

核心指标（全量，2025-12-08）：
| 模型 | IC | ICIR | IC>0 | LS Sharpe | MaxDD | Turnover | 文件 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | 0.0361 | 0.2706 | 62.96% | 0.8956 | 0.6287 | 0.7686 | `results/transformer_oldpca/transformer/stats_20251208_111045.json` |
| TFA 优化版 | 0.0218 | 0.2450 | 59.26% | 0.7905 | 0.4785 | 0.8330 | `results/tfa_oldpca/stats_20251208_125607.json` |
| Ridge | 0.0212 | 0.1854 | 62.96% | 0.8224 | 0.9953 | 0.8565 | `report/11pca/stats/ridge_stats_oldpca.json` |
| MLP | -0.0020 | -0.0210 | 50.62% | -0.0756 | 1.0073 | 0.8271 | `report/11pca/stats/mlp_stats_oldpca.json` |

消融（11 因子，rank，max_prediction_dates=20 抽样）：
| 设置 | IC | ICIR | LS Sharpe | MaxDD | 备注 |
| --- | --- | --- | --- | --- | --- |
| 去掉重构（alpha=0） | 0.0352 | 0.3085 | 0.3556 | 0.4518 | 预测显著提升，Sharpe 下降 |
| 去掉平滑/正交（beta=0,gamma=0） | 0.0094 | 0.0697 | 0.4905 | 0.5384 | 预测变差，不建议 |
| alpha=0.01 | 待跑 | 待跑 | 待跑 | 待跑 | 占位 |
| alpha=0.02 | 待跑 | 待跑 | 待跑 | 待跑 | 占位 |

要点：
- 少因子信噪比更高，Transformer 在预测上领先，TFA 回撤更小。
- 重构权重（alpha）过大抑制预测；减小 alpha 可望提升 IC/ICIR，需权衡 Sharpe。
- beta/gamma 提供稳定性，去掉会明显恶化预测。
- 长-only Sharpe 异常高需复核。***
