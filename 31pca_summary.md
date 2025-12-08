# 31 PCA 因子结果概览（2025-12-08）

输入与设置：
- 特征：`feature/pca_feature_store.csv`（31 个 PCA 因子），序列长度 36，train/val 窗口 60/12，rank-based 目标（每月横截面分位归一化），returns 来自 `data/cleaned/monthly_returns_proxy.csv`，交易成本 30 bps。
- 输出目录：`more_pca_results/`。

已完成模型（全量日期）：
| 模型 | 目标 | 核心配置 | 指标（全样本） | 文件 |
| --- | --- | --- | --- | --- |
| Ridge | 标准化特征，rank 目标 | α=1.0 | IC 0.0144，ICIR 0.1351，IC>0 56.8%，LS Sharpe 0.3709，MaxDD 0.6870，Turnover 0.8781 | `more_pca_results/ridge/stats_20251207_234935.json` |
| Random Forest | 标准化特征，rank 目标 | n_estimators=100, max_depth=10, min_samples_split=10 | IC 0.0361，ICIR 0.2854，IC>0 62.96%，LS Sharpe 0.5787，MaxDD 0.7528，Turnover 0.7043 | `more_pca_results/random_forest/predictions_20251208_005818.csv` (+performance PNG) |
| TFA 优化版 | rank 目标 | d_model=64, heads=4, enc_layers=2, latent=3, lr=5e-4, alpha/beta/gamma=0.05/0.01/0.005, epochs=50 | IC 0.0141，ICIR 0.1288，IC>0 59.26%，LS Sharpe 0.0728，MaxDD 0.7364，Turnover 0.8730；Long Sharpe 23.29（需复核） | `more_pca_results/tfa/predictions_20251208_022440.csv`，`performance_20251208_022440.png` |

待补模型（尚未生成结果）：
- Transformer（31 因子）：未跑。建议命令：`python train.py --model transformer --device auto --pca_path feature/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir more_pca_results`
- MLP（31 因子）：未跑；现已修正输入维度为 31*36。命令同上将 `--model mlp`。

提示：
- TFA 长-only Sharpe 极高，需用回测 notebook 或独立校验确认。
- 如需抽样或节省显存，可在命令里加 `--max_prediction_dates 20` 或调小 `--batch_size`。***
