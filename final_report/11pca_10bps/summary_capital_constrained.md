# 11pca Capital-Constrained Metrics (10bps per side)

- Parameters: long_pct=0.1, short_pct=0.1, transaction_cost=0.001, long_weight=0.5, short_weight=0.5, min_ls_return=-0.999

## transformer

- IC_mean: 0.0361
- IC_IR: 0.2706
- IC>0: 62.96%
- LS Sharpe: 0.9348
- MaxDD: 0.3613
- Turnover: 0.7686

## tfa

- IC_mean: 0.0218
- IC_IR: 0.2450
- IC>0: 59.26%
- LS Sharpe: 0.8434
- MaxDD: 0.2588
- Turnover: 0.8330

## random_forest

- IC_mean: 0.0481
- IC_IR: 0.2574
- IC>0: 50.00%
- LS Sharpe: 1.0070
- MaxDD: 0.1573
- Turnover: 0.6819

## tfa_ablate_alpha0

- IC_mean: 0.0352
- IC_IR: 0.3085
- IC>0: 60.00%
- LS Sharpe: 0.4023
- MaxDD: 0.2412
- Turnover: 0.8623

## tfa_ablate_bg0

- IC_mean: 0.0094
- IC_IR: 0.0697
- IC>0: 60.00%
- LS Sharpe: 0.5242
- MaxDD: 0.2908
- Turnover: 0.8599

## tfa_alpha0p01

- IC_mean: 0.0030
- IC_IR: 0.0255
- IC>0: 50.00%
- LS Sharpe: 0.2772
- MaxDD: 0.3146
- Turnover: 0.8315

## tfa_alpha0p02

- IC_mean: 0.0186
- IC_IR: 0.1655
- IC>0: 60.00%
- LS Sharpe: 0.8759
- MaxDD: 0.1239
- Turnover: 0.8586
## ridge

- IC_mean: 0.0310
- IC_IR: 0.2534
- IC>0: 61.73%
- LS Sharpe: 0.9459
- MaxDD: 0.3045
- Turnover: 0.8442

## mlp

- IC_mean: -0.0056
- IC_IR: -0.0614
- IC>0: 44.44%
- LS Sharpe: 0.0674
- MaxDD: 0.2509
- Turnover: 0.8359
