#!/usr/bin/env python3
"""
GPU环境测试脚本
测试PyTorch和MPS (Apple Silicon GPU)是否正常工作
"""

import torch
import sys
import time

def main():
    print("="*60)
    print("Alpha-Hunter GPU 环境测试")
    print("="*60)
    
    # 基本信息
    print(f"\n📦 环境信息:")
    print(f"  Python版本: {sys.version.split()[0]}")
    print(f"  PyTorch版本: {torch.__version__}")
    
    # MPS检查
    print(f"\n🍎 Apple Silicon GPU (MPS):")
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    
    print(f"  MPS可用: {mps_available}")
    print(f"  MPS已构建: {mps_built}")
    
    if mps_available:
        print("\n✅ GPU加速已启用!")
        print("   建议使用: --device mps 或 --device auto")
        
        # 速度测试
        print("\n🏃 性能测试中...")
        size = 1000
        n_iterations = 10
        
        # CPU测试
        x_cpu = torch.randn(size, size)
        start = time.time()
        for _ in range(n_iterations):
            y_cpu = torch.matmul(x_cpu, x_cpu)
        cpu_time = time.time() - start
        
        # MPS测试
        try:
            x_mps = torch.randn(size, size, device='mps')
            torch.mps.synchronize()
            start = time.time()
            for _ in range(n_iterations):
                y_mps = torch.matmul(x_mps, x_mps)
            torch.mps.synchronize()
            mps_time = time.time() - start
            
            speedup = cpu_time / mps_time
            
            print(f"\n📊 性能结果 ({n_iterations}次矩阵乘法):")
            print(f"  CPU时间: {cpu_time:.4f}s")
            print(f"  MPS时间: {mps_time:.4f}s")
            print(f"  加速倍数: {speedup:.2f}x 🚀")
            
            if speedup > 2:
                print(f"\n🎉 GPU加速效果显著!")
            elif speedup > 1:
                print(f"\n✅ GPU加速正常")
            else:
                print(f"\n⚠️  GPU似乎没有加速效果，检查系统设置")
        
        except Exception as e:
            print(f"\n❌ MPS测试失败: {e}")
            print("   尝试更新PyTorch或使用CPU")
    
    else:
        print("\n⚠️  MPS不可用")
        if not mps_built:
            print("   PyTorch没有编译MPS支持")
            print("   解决方案: pip install --pre torch torchvision")
        else:
            print("   你的设备可能不支持MPS")
            print("   这不影响功能，只是速度较慢")
    
    # 设备推荐
    print("\n🎯 训练建议:")
    if mps_available:
        print("  python train_tfa.py --device mps --batch_size 256")
    else:
        print("  python train_tfa.py --device cpu --batch_size 128")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)

if __name__ == '__main__':
    main()

