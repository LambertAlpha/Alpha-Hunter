# 11 因子模型指标汇总（rank 目标）

## 主模型
- Transformer: `results/transformer_oldpca/transformer/stats_20251208_111045.json`
```
IC_mean: 0.0361
IC_IR: 0.2706
IC>0: 62.96%
LS Sharpe: 0.8956
MaxDD: 0.6287
Turnover: 0.7686
```
- TFA 优化版: `results/tfa_oldpca/stats_20251208_125607.json`
```
IC_mean: 0.0218
IC_IR: 0.2450
IC>0: 59.26%
LS Sharpe: 0.7905
MaxDD: 0.4785
Turnover: 0.8330
```
- Ridge: `report/11pca/stats/ridge_stats_oldpca.json`
```
IC_mean: 0.0212
IC_IR: 0.1854
IC>0: 62.96%
LS Sharpe: 0.8224
MaxDD: 0.9953
Turnover: 0.8565
```
- MLP: `report/11pca/stats/mlp_stats_oldpca.json`
```
IC_mean: -0.0020
IC_IR: -0.0210
IC>0: 50.62%
LS Sharpe: -0.0756
MaxDD: 1.0073
Turnover: 0.8271
```

## TFA 消融（max_prediction_dates=20 抽样）
- 去掉重构 (alpha=0): `report/11pca/stats/tfa_ablate_alpha0.json`
```
IC_mean: 0.0352
IC_IR: 0.3085
IC>0: 60.0%
LS Sharpe: 0.3556
MaxDD: 0.4518
Turnover: 0.8623
```
- 去掉平滑/正交 (beta=0, gamma=0): `report/11pca/stats/tfa_ablate_bg0.json`
```
IC_mean: 0.0094
IC_IR: 0.0697
IC>0: 60.0%
LS Sharpe: 0.4905
MaxDD: 0.5384
Turnover: 0.8599
```
- alpha=0.01: `results/tfa_oldpca_alpha0p01/predictions_20251208_181818.csv` (计算得)
```
IC_mean: 0.0030
IC_IR: 0.0255
IC>0: 50.0%
LS Sharpe: 0.2381
MaxDD: 0.5563
Turnover: 0.8315
```
- alpha=0.02: `results/tfa_oldpca_alpha0p02/predictions_20251208_181930.csv` (计算得)
```
IC_mean: 0.0186
IC_IR: 0.1655
IC>0: 60.0%
LS Sharpe: 0.8334
MaxDD: 0.2473
Turnover: 0.8586
```
