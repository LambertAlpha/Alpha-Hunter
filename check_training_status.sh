#!/bin/bash
# 检查 TFA 训练状态的脚本

echo "🔍 检查训练状态..."
echo ""

# 1. 检查进程
echo "1️⃣  检查训练进程:"
if ps aux | grep -i "train_tfa.py" | grep -v grep > /dev/null; then
    echo "   ✅ 训练进程正在运行"
    ps aux | grep -i "train_tfa.py" | grep -v grep | awk '{print "   PID:", $2, "CPU:", $3"%", "内存:", $4"%"}'
else
    echo "   ❌ 没有训练进程在运行"
fi
echo ""

# 2. 检查最新的日志文件
echo "2️⃣  最新日志文件:"
LATEST_LOG=$(ls -t results/tfa/train_tfa_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   📄 $LATEST_LOG"
    echo "   📊 最后10行:"
    tail -10 "$LATEST_LOG" | sed 's/^/      /'
    echo ""
    
    # 检查是否有完成信息
    if grep -q "Training completed successfully" "$LATEST_LOG"; then
        echo "   ✅ 训练已完成！"
    elif grep -q "Performance Summary" "$LATEST_LOG"; then
        echo "   ✅ 训练已完成（正在评估）"
    elif grep -q "Generated.*predictions" "$LATEST_LOG"; then
        echo "   ⏳ 训练完成，正在评估性能..."
    else
        echo "   ⏳ 训练进行中..."
    fi
else
    echo "   ❌ 没有找到日志文件"
fi
echo ""

# 3. 检查结果文件
echo "3️⃣  最新结果文件:"
LATEST_CSV=$(ls -t results/tfa/predictions_*.csv 2>/dev/null | head -1)
LATEST_JSON=$(ls -t results/tfa/stats_*.json 2>/dev/null | head -1)
LATEST_PNG=$(ls -t results/tfa/performance_*.png 2>/dev/null | head -1)

if [ -n "$LATEST_CSV" ]; then
    echo "   ✅ 预测文件: $LATEST_CSV"
    echo "      大小: $(ls -lh "$LATEST_CSV" | awk '{print $5}')"
    echo "      时间: $(ls -lh "$LATEST_CSV" | awk '{print $6, $7, $8}')"
else
    echo "   ⏳ 预测文件尚未生成"
fi

if [ -n "$LATEST_JSON" ]; then
    echo "   ✅ 统计文件: $LATEST_JSON"
else
    echo "   ⏳ 统计文件尚未生成"
fi

if [ -n "$LATEST_PNG" ]; then
    echo "   ✅ 性能图表: $LATEST_PNG"
else
    echo "   ⏳ 性能图表尚未生成"
fi
echo ""

# 4. 如果训练完成，显示关键指标
if [ -n "$LATEST_JSON" ]; then
    echo "4️⃣  关键性能指标:"
    python3 << 'PYTHON'
import json
import sys
from pathlib import Path

json_file = Path("results/tfa")
json_files = sorted(json_file.glob("stats_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

if json_files:
    with open(json_files[0]) as f:
        stats = json.load(f)
    
    print(f"   📊 IC Mean: {stats.get('ic_mean', 'N/A'):.4f}" if isinstance(stats.get('ic_mean'), (int, float)) else f"   📊 IC Mean: {stats.get('ic_mean', 'N/A')}")
    print(f"   📊 ICIR: {stats.get('icir', 'N/A'):.4f}" if isinstance(stats.get('icir'), (int, float)) else f"   📊 ICIR: {stats.get('icir', 'N/A')}")
    print(f"   📊 Sharpe: {stats.get('ls_sharpe', 'N/A'):.4f}" if isinstance(stats.get('ls_sharpe'), (int, float)) else f"   📊 Sharpe: {stats.get('ls_sharpe', 'N/A')}")
else:
    print("   ⏳ 统计文件尚未生成")
PYTHON
fi

echo ""
echo "💡 提示: 运行 'tail -f $LATEST_LOG' 可以实时查看训练进度"
