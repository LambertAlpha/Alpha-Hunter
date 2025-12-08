# 复现命令

## 31 因子（feature/pca_feature_store.csv，rank 目标）
- Transformer：`python train.py --model transformer --device auto --pca_path feature/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir more_pca_results`
- MLP：`python train.py --model mlp --device auto --pca_path feature/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir more_pca_results`
- Ridge：`python train.py --model ridge --device auto --pca_path feature/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir more_pca_results`
- Random Forest：`python train.py --model random_forest --device auto --pca_path feature/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir more_pca_results`
- TFA 优化版：`python train_tfa.py --device auto --pca_path feature/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir more_pca_results/tfa`

## 11 因子（data/pca/old/pca_feature_store.csv，rank 目标）
- Transformer：`python train.py --model transformer --device auto --pca_path data/pca/old/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir results/transformer_oldpca`
- TFA 优化版：`python train_tfa.py --device auto --pca_path data/pca/old/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir results/tfa_oldpca`

## TFA 消融（11 因子，抽样 20 期示例，可去掉 max_prediction_dates 全量）
- 去掉重构：`python train_tfa.py --device auto --pca_path data/pca/old/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir results/tfa_oldpca_ablate_alpha0 --alpha 0 --max_prediction_dates 20`
- 去掉平滑/正交：`python train_tfa.py --device auto --pca_path data/pca/old/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir results/tfa_oldpca_ablate_bg0 --beta 0 --gamma 0 --max_prediction_dates 20`
- 降低重构权重（示例 alpha=0.01）：`python train_tfa.py --device auto --pca_path data/pca/old/pca_feature_store.csv --returns_path data/cleaned/monthly_returns_proxy.csv --output_dir results/tfa_oldpca_alpha0p01 --alpha 0.01 --beta 0.01 --gamma 0.005 --max_prediction_dates 20`
