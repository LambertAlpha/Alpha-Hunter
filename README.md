# Alpha-Hunter

时序因子自编码器（TFA）用于股票收益预测

## 快速开始

### 训练模型

```bash
# TFA模型
python train_tfa.py --epochs 50

# 基准模型
python train.py --model ridge
python train.py --model transformer
```

### 实验跟踪

```python
from src.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

# 记录实验
exp_dir = tracker.start_experiment(
    name='tfa_baseline',
    description='TFA基准',
    config={'d_model': 128},
    tags=['tfa']
)

# 记录指标
tracker.log_metrics(exp_dir, {'IC_mean': 0.067})
tracker.finish_experiment(exp_dir)

# 对比和导出
df = tracker.compare_experiments()
latex = tracker.export_to_latex(df)  # LaTeX表格
```

## 项目结构

```
src/
├── nn_utils.py              # 共享组件
├── models.py                # 基准模型
├── models_tfa.py            # TFA模型
├── data_loader.py           # 数据加载（向量化）
├── trainer.py               # 训练框架（缓存）
├── evaluator.py             # 评估
├── experiment_tracker.py    # 实验跟踪
└── config.py                # 配置

train_tfa.py                 # TFA训练
train.py                     # 基准训练
```

## 核心优化

- 向量化数据加载（10-50倍提升）
- 验证集缓存
- 实验自动跟踪

详见 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)

## 测试

```bash
python -m src.test_optimizations
```

## 团队

Lin Boyi, Qian Linyi, Yan Tingyu
香港中文大学（深圳）
