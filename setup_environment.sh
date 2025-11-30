#!/bin/bash
# Alpha-Hunter 环境配置脚本
# 适用于 Apple Silicon Mac (M1/M2/M3)

echo "🚀 开始配置 Alpha-Hunter 环境..."
echo ""

# 1. 创建conda环境
echo "📦 Step 1: 创建 conda 环境 'ml'..."
conda env create -f environment.yml

# 2. 激活环境
echo ""
echo "✅ Step 2: 激活环境..."
conda activate ml

# 3. 安装PyTorch (Apple Silicon优化版本)
echo ""
echo "🔥 Step 3: 安装 PyTorch with MPS support..."
pip install torch torchvision torchaudio

# 4. 验证安装
echo ""
echo "🧪 Step 4: 验证安装..."
python -c "
import torch
import sys
print('✅ Python version:', sys.version)
print('✅ PyTorch version:', torch.__version__)
print('✅ MPS (GPU) available:', torch.backends.mps.is_available())
print('✅ MPS built:', torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    print('🎉 GPU加速已启用! (使用Apple Metal)')
else:
    print('⚠️  GPU不可用，将使用CPU')
"

echo ""
echo "🎓 安装完成！"
echo ""
echo "下一步："
echo "  1. 激活环境: conda activate ml"
echo "  2. 测试TFA:   jupyter notebook notebooks/TFA_Demo.ipynb"
echo "  3. 完整训练:   python train_tfa.py --device mps"
echo ""

