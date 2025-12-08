# report 目录导览

- **31pca/**: 31 因子版本（feature/pca_feature_store.csv），rank 目标，含各模型指标与图表。
- **11pca/**: 11 因子版本（data/pca/old/pca_feature_store.csv），rank 目标，含各模型指标与图表（含消融）。
- **comparisons/**: 31 vs 11 的指标对比与结论。
- **commands.md**: 复现训练/消融的命令。

注意事项：
- 长-only Sharpe 偏高的结果需回测/校验，不应直接作为交易结论。
- 消融实验标注在 11 因子下：alpha=0 提升 IC/ICIR 但 Sharpe 下降；去掉 beta/gamma 预测变差。***
