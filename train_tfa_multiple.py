"""
运行多次TFA训练，测试不同参数配置
"""

import subprocess
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# 参数配置列表
CONFIGS = [
    {
        "name": "config_a_conservative",
        "description": "保守配置 - 防止过拟合",
        "params": {
            "--d_model": "64",
            "--n_heads": "4",
            "--n_encoder_layers": "2",
            "--n_latent_factors": "3",
            "--dropout": "0.2",
            "--lr": "0.0005",
            "--weight_decay": "0.001",
            "--epochs": "100",
            "--alpha": "0.05",
            "--beta": "0.02",
            "--gamma": "0.01",
            "--early_stopping_patience": "10",
        }
    },
    {
        "name": "config_b_balanced",
        "description": "平衡配置 - 推荐",
        "params": {
            "--d_model": "96",
            "--n_heads": "6",
            "--n_encoder_layers": "3",
            "--n_latent_factors": "4",
            "--dropout": "0.15",
            "--lr": "0.001",
            "--weight_decay": "0.0005",
            "--epochs": "80",
            "--alpha": "0.08",
            "--beta": "0.03",
            "--gamma": "0.01",
            "--early_stopping_patience": "8",
        }
    },
    {
        "name": "config_c_aggressive",
        "description": "激进配置 - 更大模型",
        "params": {
            "--d_model": "128",
            "--n_heads": "8",
            "--n_encoder_layers": "4",
            "--n_latent_factors": "5",
            "--dropout": "0.1",
            "--lr": "0.0015",
            "--weight_decay": "0.0001",
            "--epochs": "100",
            "--alpha": "0.1",
            "--beta": "0.05",
            "--gamma": "0.01",
            "--early_stopping_patience": "10",
        }
    },
    {
        "name": "config_d_focused",
        "description": "聚焦配置 - 降低重构权重，专注预测",
        "params": {
            "--d_model": "96",
            "--n_heads": "6",
            "--n_encoder_layers": "3",
            "--n_latent_factors": "4",
            "--dropout": "0.15",
            "--lr": "0.001",
            "--weight_decay": "0.0005",
            "--epochs": "80",
            "--alpha": "0.03",  # 降低重构权重
            "--beta": "0.02",
            "--gamma": "0.01",
            "--early_stopping_patience": "8",
        }
    },
]

def run_training(config):
    """运行单次训练"""
    print(f"\n{'='*60}")
    print(f"Training: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}\n")
    
    # 构建命令
    cmd = ["python", "train_tfa.py", "--verbose"]
    
    # 添加参数（注意：train_tfa.py可能不支持所有参数，需要检查）
    for key, value in config['params'].items():
        # 移除--前缀，因为train_tfa.py的参数名可能不同
        param_name = key.replace("--", "")
        # 检查是否是train_tfa.py支持的参数
        if param_name in ["d_model", "n_heads", "n_encoder_layers", "n_latent_factors",
                          "lr", "epochs", "batch_size", "alpha", "beta", "gamma"]:
            cmd.extend([key, value])
    
    # 运行训练
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)
        return False, e.stderr

def collect_results():
    """收集所有训练结果"""
    results_dir = Path("results/tfa")
    results = []
    
    # 查找所有stats文件
    for stats_file in results_dir.glob("stats_*.json"):
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # 提取关键指标
            results.append({
                "config": stats_file.stem.replace("stats_", ""),
                "IC_mean": stats.get("IC_mean", None),
                "IC_IR": stats.get("IC_IR", None),
                "LS_sharpe": stats.get("LS_sharpe", None),
                "LS_mean_return": stats.get("LS_mean_return", None),
                "LS_max_drawdown": stats.get("LS_max_drawdown", None),
            })
        except Exception as e:
            print(f"Error reading {stats_file}: {e}")
    
    return results

def main():
    print("="*60)
    print("TFA Multiple Training Experiments")
    print("="*60)
    
    # 运行所有配置
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] Starting {config['name']}...")
        success, output = run_training(config)
        
        if not success:
            print(f"⚠️  Training failed for {config['name']}")
            continue
    
    # 收集结果
    print("\n" + "="*60)
    print("Collecting Results...")
    print("="*60)
    
    results = collect_results()
    
    if results:
        df = pd.DataFrame(results)
        print("\n📊 Results Summary:")
        print(df.to_string(index=False))
        
        # 保存结果
        summary_file = Path("results/tfa") / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(summary_file, index=False)
        print(f"\n✅ Summary saved to {summary_file}")
        
        # 找出最佳配置
        if len(df) > 0:
            best_ic = df.loc[df['IC_mean'].idxmax()]
            best_sharpe = df.loc[df['LS_sharpe'].idxmax()]
            
            print("\n🏆 Best Configurations:")
            print(f"  Best IC: {best_ic['config']} (IC={best_ic['IC_mean']:.4f})")
            print(f"  Best Sharpe: {best_sharpe['config']} (Sharpe={best_sharpe['LS_sharpe']:.4f})")
    else:
        print("⚠️  No results found")

if __name__ == "__main__":
    main()
