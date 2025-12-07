"""快速测试优化后的代码"""

import sys
from pathlib import Path

# Test 1: 测试共享模块导入
print("测试优化...")
try:
    from .nn_utils import PositionalEncoding
    from .models import TransformerEncoder
    from .models_tfa import TemporalFactorAutoencoder
    print("✓ 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# Test 2: 测试数据加载
try:
    from .data_loader import SequenceDataLoader
    pca_path = Path(__file__).parent.parent / 'feature/pca_feature_store.csv'

    if pca_path.exists():
        loader = SequenceDataLoader(pca_path, sequence_length=36)
        if len(loader.dates) > 36:
            result = loader.build_sequences(loader.dates[50], return_dict=True)
            print(f"✓ 数据加载: {result['X'].shape}")
    else:
        print("⚠ PCA文件不存在，跳过数据测试")
except Exception as e:
    print(f"✗ 数据加载失败: {e}")

# Test 3: 测试实验跟踪器
try:
    from .experiment_tracker import ExperimentTracker
    import shutil

    tracker = ExperimentTracker(base_dir='test_exp')
    exp_dir = tracker.start_experiment('test', 'test', {}, ['test'])
    tracker.log_metrics(exp_dir, {'test': 1.0})
    tracker.finish_experiment(exp_dir)
    shutil.rmtree('test_exp')
    print("✓ 实验跟踪器正常")
except Exception as e:
    print(f"✗ 实验跟踪器失败: {e}")

print("\n所有测试完成")
