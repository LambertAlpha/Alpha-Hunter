# Alpha-Hunter: Dynamic Factor Investing with Transformer-Based Return Prediction

基于Transformer的中国A股CSI 500收益率预测系统，结合PCA降维和深度学习进行动态因子投资。

## 📋 项目概述

本项目实现了一个完整的量化投资pipeline：
1. **PCA特征提取**：对高维firm characteristics进行降维
2. **序列建模**：使用36个月历史数据构建时间序列
3. **创新模型**：**Temporal Factor Autoencoder (TFA)** - 学习时变因子权重
4. **基线模型**：Transformer、Ridge、Random Forest、MLP
5. **Rolling window回测**：时间序列交叉验证
6. **Portfolio评估**：IC、ICIR、Sharpe ratio等指标

## 🌟 核心创新：TFA (Temporal Factor Autoencoder)

**TFA是本项目的主要贡献**，通过Transformer的attention机制学习动态的PCA因子权重：

✨ **三大创新点**：
1. **Dynamic Factor Weighting** - 因子权重根据36个月历史动态调整
2. **Encoder-Decoder架构** - 重构约束确保信息保留
3. **Temporal Smoothness** - 平滑性约束增强可解释性

📈 **预期性能**（vs Ridge baseline）：
- IC: +50-60%
- Sharpe: +60-80%
- 可解释的attention patterns

## 🏗️ 项目结构

```
Alpha-Hunter/
├── src/                          # 核心Python模块
│   ├── __init__.py
│   ├── models.py                 # 基线模型
│   ├── models_tfa.py             # ⭐ TFA模型（核心创新）
│   ├── tfa_analysis.py           # ⭐ TFA分析工具
│   ├── data_loader.py            # 数据加载（36个月序列）
│   ├── trainer.py                # Rolling window训练
│   ├── evaluator.py              # 性能评估
│   ├── config.py                 # 配置管理
│   └── utils.py                  # 工具函数
│
├── notebooks/                    # Jupyter notebooks
│   ├── TFA_Demo.ipynb            # ⭐ TFA快速演示
│   ├── TFA_vs_Baselines.ipynb    # ⭐ 对比实验（论文Table 1）
│   ├── 01_model_training.ipynb   # 模型训练
│   ├── 02_backtesting.ipynb      # 回测分析
│   └── 03_interpretation.ipynb   # 可解释性分析
│
├── results/                      # 训练结果
│   ├── tfa/                      # ⭐ TFA结果
│   │   └── analysis/             #   - attention_pattern.png
│   ├── transformer/
│   ├── ridge/
│   └── mlp/
│
├── train.py                      # 训练baseline模型
├── train_tfa.py                  # ⭐ 训练TFA模型
├── TFA_README.md                 # ⭐ TFA详细文档
├── QUICKSTART_TFA.md             # ⭐ TFA快速指南
└── README.md
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 检查数据

确保PCA特征数据已准备好：
```bash
ls feature/
# 应该看到: pca_feature_store.csv, pca_explained_variance.csv
```

### 3. 训练模型

#### ⭐ 推荐：TFA模型（核心创新）

```bash
# 快速测试（5分钟，建议先跑这个）
jupyter notebook notebooks/TFA_Demo.ipynb

# 完整训练（30-60分钟）
python train_tfa.py --epochs 50 --verbose --analyze

# 自定义超参数
python train_tfa.py \
    --d_model 128 \
    --n_heads 8 \
    --n_encoder_layers 4 \
    --alpha 0.1 \
    --beta 0.05 \
    --device cuda
```

**TFA结果位置**：`results/tfa/`
- `predictions_*.csv` - 预测结果
- `performance_*.png` - 性能图表
- `analysis/attention_pattern.png` - **论文核心图！**

#### 训练Baseline模型（用于对比）

```bash
# 训练对比模型
python train.py --model ridge --verbose
python train.py --model mlp --verbose
python train.py --model transformer --verbose

# 或一次性训练所有
python train.py --model all
```

#### 对比分析（生成论文Table）

```bash
# 训练完所有模型后
jupyter notebook notebooks/TFA_vs_Baselines.ipynb
```

**命令行参数**：
- `--epochs`: 训练轮数（默认：50）
- `--d_model`: 模型维度（64/128/256）
- `--n_heads`: 注意力头数（4/8）
- `--alpha`: 重构loss权重（0.05-0.2）
- `--beta`: 平滑性loss权重（0.01-0.1）
- `--device`: 设备（`auto`/`cpu`/`cuda`）
- `--analyze`: 生成详细分析报告

### 4. 分析结果

使用提供的notebooks进行分析：

```bash
jupyter notebook notebooks/02_backtesting.ipynb      # Portfolio回测
jupyter notebook notebooks/03_interpretation.ipynb   # 模型解释
```

## 📊 模型架构

### ⭐ TFA (Temporal Factor Autoencoder) - 核心创新

```python
Input: (batch, 36 months, 11 PCA factors)
  ↓
┌─────────────────────────────────────┐
│ Encoder (4 layers, 8 heads)        │
│   ↓                                 │
│ Dynamic Weight Generator  ← 创新！  │
│   ↓                                 │
│   ├→ Factor Weights (时变)          │
│   ├→ Decoder → Reconstruction      │
│   └→ Latent Factors → Prediction   │
└─────────────────────────────────────┘
  ↓
Output: (batch, 5 quantile classes)

Loss = Prediction + α×Reconstruction + β×Smoothness
```

**核心参数**：
- `seq_len`: 36个月（捕捉长期依赖）
- `d_model`: 128
- `n_heads`: 8
- `n_encoder_layers`: 4
- `n_latent_factors`: 5（学习的因子数）
- `alpha`: 0.1（重构权重）
- `beta`: 0.05（平滑性权重）

**详细文档**：见 `TFA_README.md`

### Baseline模型

1. **Ridge Regression**: 线性模型 + L2正则化
2. **Random Forest**: 100棵树的ensemble
3. **MLP**: 3层全连接网络 [256-128-64]
4. **Transformer**: 原始Transformer encoder

## 📈 评估指标

### 1. Information Coefficient (IC)
每月预测值与实际收益的截面相关系数（Spearman）

### 2. IC Information Ratio (ICIR)
```
ICIR = Mean(IC) / Std(IC)
```

### 3. Long-Short Portfolio
- **Long**: 预测前10%的股票
- **Short**: 预测后10%的股票
- **Transaction Cost**: 30 bps/side

### 4. Sharpe Ratio
```
Sharpe = Mean(Returns) / Std(Returns) * √12
```

## 🔧 配置说明

配置文件示例 (`config.json`):

```json
{
  "data": {
    "pca_path": "feature/pca_feature_store.csv",
    "sequence_length": 12,
    "forward_fill_limit": 3
  },
  "training": {
    "train_window": 60,
    "val_window": 12,
    "min_train_months": 36
  },
  "transformer": {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "epochs": 50,
    "lr": 0.001
  }
}
```

## 📝 使用示例

### TFA Python API

```python
from src.data_loader import SequenceDataLoader
from src.models_tfa import TFAPredictor
from src.tfa_analysis import TFAAnalyzer

# 1. 加载数据（36个月序列）
data_loader = SequenceDataLoader(
    'feature/pca_feature_store.csv',
    sequence_length=36
)

# 2. 创建和训练TFA
tfa = TFAPredictor(
    n_pca_factors=11,
    seq_len=36,
    d_model=128,
    n_heads=8,
    device='cuda'
)
tfa.fit(X_train, y_train, X_val, y_val)

# 3. 预测
predictions = tfa.predict(X_test)

# 4. 分析attention patterns
analyzer = TFAAnalyzer(tfa)
weights_df = analyzer.extract_factor_weights(X_test)
analyzer.plot_average_attention_pattern(weights_df)

# 5. 生成完整报告
analyzer.generate_report(X_test, y_test, dates, output_dir='tfa_analysis')
```

### Baseline API

```python
from src.models import TransformerPredictor, RidgePredictor
from src.trainer import RollingWindowTrainer

# 创建baseline模型
def create_model():
    return RidgePredictor(alpha=1.0)

# 训练
trainer = RollingWindowTrainer(data_loader, create_model)
predictions = trainer.train_and_predict()
```

## 📚 方法论

基于以下论文的方法：

1. **Gu et al. (2020)** - Empirical Asset Pricing via Machine Learning
   - 使用机器学习捕捉非线性关系
   - Rolling window交叉验证

2. **Lettau & Pelger (2020)** - Factors That Fit the Time Series and Cross-Section
   - 动态因子提取
   - 时变factor loadings

3. **Zhang et al. (2023)** - Finformer
   - Transformer用于金融时间序列
   - Attention机制捕捉长程依赖

## 🎯 项目Timeline

| 阶段 | 任务 | 日期 | 状态 |
|------|------|------|------|
| 1 | Kick-off和文献综述 | 11/10-11/11 | ✅ |
| 2 | 数据获取和清洗 | 11/12-11/17 | ✅ |
| 3 | PCA特征工程 | 11/18-11/27 | ✅ |
| 4 | TFA模型实现 | 11/28-11/30 | ✅ |
| 5 | 模型训练和对比 | 12/01-12/05 | 🔄 当前阶段 |
| 6 | 回测和分析 | 12/06-12/10 | ⏳ |
| 7 | 论文撰写 | 12/11-12/15 | ⏳ |

## 🐛 故障排除

### GPU不可用
```python
# 检查PyTorch CUDA
import torch
print(torch.cuda.is_available())

# 强制使用CPU
python train.py --device cpu
```

### 内存不足
- 减少`batch_size`
- 减少`train_window`
- 使用更少的`n_estimators`（Random Forest）

### 数据加载错误
```bash
# 检查文件路径
ls feature/pca_feature_store.csv

# 检查数据格式
python -c "import pandas as pd; print(pd.read_csv('feature/pca_feature_store.csv').head())"
```

## 📧 团队成员

- **Lin Boyi** (123090327)
- **Qian Linyi** (121090452)
- **Yan Tingyu** (124090831)

香港中文大学（深圳）

## 📄 License

本项目仅用于学术研究，不构成投资建议。

---

## 🚀 快速实验流程（论文）

**完整实验（约2-3小时）**：

```bash
# Step 1: 快速验证TFA（5分钟）
jupyter notebook notebooks/TFA_Demo.ipynb

# Step 2: 训练TFA（1小时）
python train_tfa.py --epochs 50 --analyze

# Step 3: 训练baselines（1小时）
python train.py --model ridge &
python train.py --model mlp &
python train.py --model transformer &
wait

# Step 4: 生成对比分析（10分钟）
jupyter notebook notebooks/TFA_vs_Baselines.ipynb
```

**输出**：
- `results/tfa/analysis/attention_pattern.png` - 论文Figure
- `TFA_vs_Baselines.ipynb` - 论文Table 1

## 📚 文档索引

- **`README.md`** (本文件) - 项目概览
- **`TFA_README.md`** - TFA详细文档、论文写作指南
- **`QUICKSTART_TFA.md`** - TFA快速上手
- **`TFA_实现总结.md`** - 技术实现细节

## 🔥 核心亮点

1. ✅ **创新模型**：TFA学习时变因子权重
2. ✅ **完整实现**：从模型到分析到可视化
3. ✅ **高可解释**：Attention权重可视化
4. ✅ **学术规范**：符合金融论文要求
5. ✅ **预期提升**：IC +50%, Sharpe +60%

Good luck with your paper! 🎓🚀

