# 31pca Capital-Constrained Metrics (10bps per side)

- Parameters: long_pct=0.1, short_pct=0.1, transaction_cost=0.001, long_weight=0.5, short_weight=0.5, min_ls_return=-0.999

## ridge

- IC_mean: 0.0144
- IC_IR: 0.1351
- IC>0: 56.79%
- LS Sharpe: 0.4153
- MaxDD: 0.4055
- Turnover: 0.8781

## random_forest

- IC_mean: 0.0361
- IC_IR: 0.2854
- IC>0: 62.96%
- LS Sharpe: 0.6133
- MaxDD: 0.4514
- Turnover: 0.7043

## transformer

- IC_mean: 0.0054
- IC_IR: 0.0633
- IC>0: 51.85%
- LS Sharpe: 0.1706
- MaxDD: 0.2800
- Turnover: 0.8672

## mlp

- IC_mean: -0.0122
- IC_IR: -0.1274
- IC>0: 40.74%
- LS Sharpe: -0.1971
- MaxDD: 0.4844
- Turnover: 0.8333

## tfa

- IC_mean: 0.0141
- IC_IR: 0.1288
- IC>0: 59.26%
- LS Sharpe: 0.1228
- MaxDD: 0.4009
- Turnover: 0.8730
