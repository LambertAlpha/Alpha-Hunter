# less_pca_summary

为后续增加 PCA 因子重训准备的报告/演示用汇总（配置、数据、输出、表现）。

## 数据与评估流程
- 特征：`feature/pca_feature_store.csv`；当前使用 11 个 PCA 因子（`n_pca_factors`=11）。
- 序列长度：36 期；缺失前向填充上限 3。
- 滚动窗口：训练 60 期，验证 12 期；最少训练 36 期。
- 目标变量两种：
  - 标准化收益（z-score）。
  - 排名收益（percentile ranks）——近期效果更好。
- 评估：多空各 10% 等权，交易成本 30 bps，年化期数 12，rf=0。

## 模型配置与最新结果（2025-12-07）

| 模型 | 目标 | 关键配置 | 输出 | 指标 |
| --- | --- | --- | --- | --- |
| Ridge（原版） | 标准化 | `alpha=1.0` | `results/ridge/predictions_20251207_013403.csv`（+PNG） | IC≈0.0212，ICIR≈0.30+，IC>0≈58%，Sharpe>0（TRAINING_RESULTS_SUMMARY.md） |
| Transformer | 排名 | d_model=64, n_heads=4, layers=2, ff=256, dropout=0.1, lr=5e-4, wd=1e-4, batch=128, epochs=50 | `results/transformer/predictions_20251207_221127.csv`、`_221450.csv`、`_023955.csv`（+性能图） | IC 0.0789，ICIR 0.3315，IC>0 60%，Sharpe 0.5864，MaxDD 38.24%（RESULTS_COMPARISON.md） |
| Transformer baseline | 标准化 | 同上架构；`train.py --model transformer` | `results/transformer_baseline/predictions.csv` | 未记录指标（仅作结构对照） |
| TFA baseline | 标准化 | d_model=128, n_heads=8, enc_layers=4, dec_layers=2, latent=5, dropout=0.1, lr=1e-3, epochs=50, alpha=0.1, beta=0.05, gamma=0.01 | `results/tfa_baseline/predictions.csv`，`performance.png` | IC≈-0.0071，ICIR<0，IC>0<50%（results/tfa_baseline/README.md） |
| TFA optimized | 排名 | d_model=64, n_heads=4, enc_layers=2, dec_layers=2, latent=3, dropout=0.1, lr=5e-4, epochs=30, alpha=0.05, beta=0.01, gamma=0.005, batch=128 | Stats: `results/tfa_optimized/stats.json` & `_20251207_114600.json`; log: `logs/tfa_optimized_20251207_102456.log` | IC 0.0189，ICIR 0.1947，IC>0 61.73%，LS Sharpe -0.1026，MaxDD 0.8938，Long Sharpe 21.39（需复核），Turnover 0.7794 |
| TFA（排名版，标准架构） | 排名 | 与 baseline 架构相同；loss 权重 alpha=0.1, beta=0.05, gamma=0.01；epochs=50 | `results/tfa/predictions_20251207_215519.csv`、`_163613.csv`、`_060040.csv`（+性能图） | 未完整汇总；在 RESULTS_COMPARISON 中用作与 Transformer 的 rank 基线对比 |
| MLP | 排名 | hidden_dims [256,128,64], dropout=0.2, lr=1e-3, wd=1e-4, batch=128, epochs=50 | `results/mlp/predictions_20251207_102928.csv`，`stats_20251207_102928.json`，`portfolio_20251207_102928.csv` | IC -0.0020，ICIR -0.0210，IC>0 50.62%，LS Sharpe -0.0756，Long Sharpe 0.0957，Turnover 0.8271 |
| MLP baseline | 标准化 | 同 MLP 架构 | `results/mlp_baseline/predictions.csv`，`stats.json`，`portfolio.csv` | 指标未整理（基线参考） |
| Random Forest | 标准化（配置） | n_estimators=100, max_depth=10, min_samples_split=10, n_jobs=-1 | 近期无产出文件 | — |

## 关键对比（表格/图表可直接引用）
- Transformer vs TFA（rank-based，同配置）：详见 `RESULTS_COMPARISON.md`
  - IC: 0.0789 vs 0.0346；ICIR: 0.3315 vs 0.3191；IC>0: 60% vs 70%
  - 风险：MaxDD 38.24% vs 24.70%，Sharpe 0.5864 vs 0.6338
  - 结论：Transformer 预测强，TFA 更稳、回撤小
- TFA 优化 vs 原版（rank-based vs standardized）：`results/tfa_optimized/stats.json` 对比 `results/tfa_baseline/README.md`
  - IC: 0.0189 vs -0.0071；ICIR: 0.1947 vs 负；IC>0: 61.73% vs <50%
  - 损失权重/架构简化+rank 目标显著改善
- MLP vs TFA/Transformer（rank-based）：`results/mlp/stats_20251207_102928.json`
  - MLP IC 近 0（-0.002），明显弱于 TFA/Transformer；可作为深度简单基线
- Ridge（standardized） vs 深度模型（rank-based）
  - Ridge IC≈0.0212，ICIR≈0.30+，是稳健线性基线；深度模型在 rank 目标下有超越空间（Transformer/TFA）

## 图表资源（报告可嵌入）
- IC 稳定性：`results/ic_stability.png`
- 预测 vs 实际散点：`results/prediction_vs_actual.png`
- PCA 相关性/解释方差/时序：`results/pca_correlation.png`，`results/pca_explained_variance.png`，`results/pca_time_series.png`
- TFA 性能图：`results/tfa/performance_20251207_060040.png`，`_163614.png`，`_215519.png`
- Transformer 性能图：`results/transformer/performance_20251207_023955.png`，`_221450.png`
- Ridge 性能图：`results/ridge/performance_20251207_013403.png`
- MLP 性能图：`results/mlp/performance_20251207_102909.png`
- TFA baseline 性能：`results/tfa_baseline/performance.png`
- 资产层分析：`results/asset_level_analysis.png`

## 报告/演示可用的简单表格模板
（可直接用 markdown/幻灯片复制）
```
| 模型 | 数据目标 | IC | ICIR | IC>0 | Sharpe | MaxDD | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | 排名 | 0.0789 | 0.3315 | 60% | 0.5864 | 38.24% | 预测最强 |
| TFA (rank) | 排名 | 0.0346 | 0.3191 | 70% | 0.6338 | 24.70% | 稳定/回撤小 |
| TFA optimized | 排名 | 0.0189 | 0.1947 | 61.73% | -0.1026 | 89.38% | 长-only Sharpe 待复核 |
| Ridge | 标准化 | 0.0212 | ~0.30+ | ~58% | >0 | — | 线性基线 |
| MLP | 排名 | -0.0020 | -0.0210 | 50.62% | -0.0756 | 100.73% | 深度弱基线 |
```

## 重训（增加 PCA 因子）提示
- 提前更新 `n_pca_factors` 与 PCA 特征文件；序列长度 36、窗口设置可沿用。
- 排名目标对 TFA 明显有效；Transformer 在排名目标下也表现强，重训建议保留两者对比。
- TFA optimized 的 Long-only Sharpe 偏高，重训后务必复核（或加回测 notebook 验证）。
- 运行脚本：`train.py --model {ridge|transformer|mlp}`；`train_tfa.py` / `train_tfa_multiple.py` 用于 TFA。默认配置见 `results/config.json`。
