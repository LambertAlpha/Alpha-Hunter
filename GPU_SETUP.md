# Mac Mini GPU 配置指南

## 🍎 Apple Silicon GPU 支持

你的 Mac Mini 使用 **Apple Silicon (M1/M2/M3)** 芯片，支持通过 **MPS (Metal Performance Shaders)** 加速深度学习！

### 硬件检查

```bash
# 查看Mac型号和芯片
system_profiler SPHardwareDataType | grep "Chip\|Model"
```

如果显示 M1/M2/M3，说明支持 GPU 加速！

---

## 🚀 环境配置

### 方法 1: 自动安装（推荐）

```bash
# 进入项目目录
cd /Users/lambertlin/Projects/Alpha-Hunter

# 运行安装脚本
bash setup_environment.sh
```

### 方法 2: 手动安装

```bash
# 1. 创建环境
conda env create -f environment.yml

# 2. 激活环境
conda activate ml

# 3. 安装PyTorch (自动支持MPS)
pip install torch torchvision torchaudio

# 4. 验证GPU
python -c "import torch; print('MPS可用:', torch.backends.mps.is_available())"
```

---

## 🎯 训练时使用GPU

### TFA训练

```bash
# 自动检测并使用MPS
python train_tfa.py --device auto --epochs 50

# 或显式指定MPS
python train_tfa.py --device mps --epochs 50
```

### Baseline训练

```bash
python train.py --model transformer --device auto
```

### Jupyter Notebook

```python
# 在notebook中检查GPU
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"MPS可用: {torch.backends.mps.is_available()}")
print(f"MPS已构建: {torch.backends.mps.is_built()}")

# 创建tensor测试
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(5, device=device)
    print(f"✅ GPU测试成功: {x.device}")
```

---

## ⚡ 性能对比

| 硬件 | 训练速度 | 推荐batch_size |
|------|----------|----------------|
| **MPS (M1/M2/M3)** | **快 3-5倍** | **128-256** |
| CPU | 慢 | 64-128 |

**建议**：
- TFA训练：使用 `--device mps --batch_size 256`
- 如果内存不足，减少 `--batch_size` 或 `--d_model`

---

## 🐛 常见问题

### Q1: "MPS backend not available"

**解决方案**：
```bash
# 重新安装PyTorch
pip uninstall torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Q2: 内存溢出 (OOM)

**解决方案**：
```bash
# 减小batch_size
python train_tfa.py --batch_size 64 --device mps

# 或减小模型
python train_tfa.py --d_model 64 --device mps
```

### Q3: 训练时卡住

**解决方案**：
```bash
# 某些操作可能不支持MPS，自动回退到CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
python train_tfa.py --device mps
```

### Q4: 想强制使用CPU

```bash
python train_tfa.py --device cpu
```

---

## 📊 性能监控

### 活动监视器

```
Spotlight搜索: 活动监视器
→ 窗口 → GPU历史记录
```

训练时应该看到GPU使用率上升！

### 命令行监控

```bash
# 安装监控工具
pip install asitop

# 运行监控
sudo asitop
```

---

## ✅ 验证安装

运行这个脚本验证一切正常：

```python
# test_gpu.py
import torch
import sys

print("="*60)
print("Alpha-Hunter GPU 环境测试")
print("="*60)
print(f"Python版本: {sys.version.split()[0]}")
print(f"PyTorch版本: {torch.__version__}")
print(f"MPS可用: {torch.backends.mps.is_available()}")
print(f"MPS已构建: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    print("\n✅ GPU加速已启用!")
    print("   建议使用: --device mps 或 --device auto")
    
    # 速度测试
    print("\n🏃 速度测试...")
    import time
    
    size = 1000
    x = torch.randn(size, size)
    
    # CPU
    start = time.time()
    y = torch.matmul(x, x)
    cpu_time = time.time() - start
    
    # MPS
    x_mps = x.to('mps')
    start = time.time()
    y_mps = torch.matmul(x_mps, x_mps)
    torch.mps.synchronize()
    mps_time = time.time() - start
    
    print(f"   CPU时间: {cpu_time:.4f}s")
    print(f"   MPS时间: {mps_time:.4f}s")
    print(f"   加速: {cpu_time/mps_time:.2f}x")
else:
    print("\n⚠️  MPS不可用，将使用CPU")
    print("   这不影响功能，只是速度较慢")

print("\n" + "="*60)
```

运行：
```bash
python test_gpu.py
```

---

## 🎓 推荐配置

### 日常开发
```bash
# 使用Jupyter，自动GPU加速
conda activate ml
jupyter notebook notebooks/TFA_Demo.ipynb
```

### 完整训练
```bash
conda activate ml
python train_tfa.py --device auto --epochs 50 --batch_size 256 --analyze
```

### 超参数搜索
```bash
# 可以并行运行多个实验
python train_tfa.py --device mps --d_model 64 &
python train_tfa.py --device mps --d_model 128 &
python train_tfa.py --device mps --d_model 256 &
```

---

**预期训练时间**（Mac Mini M1/M2）：
- TFA Demo (小样本): ~2-3分钟
- TFA完整训练: ~20-40分钟（vs CPU: 1-2小时）
- 所有模型对比: ~1小时（vs CPU: 3-4小时）

🚀 享受GPU加速的快感！

