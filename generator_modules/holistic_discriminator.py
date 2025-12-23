"""
Holistic Discriminator Loader for generator_modules
整体判别器加载器 - 从adult_v2加载已训练的判别器
"""

import sys
import os

# 添加adult_v2到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
adult_v2_path = os.path.join(parent_dir, 'adult_v2')
if adult_v2_path not in sys.path:
    sys.path.insert(0, adult_v2_path)

try:
    from adult_holistic_discriminator import AdultHolisticDiscriminator, train_holistic_discriminator
    HAS_DISCRIMINATOR = True
except ImportError as e:
    print(f"⚠ 无法加载整体判别器: {e}")
    print("  请先运行: python train_holistic_discriminator.py")
    HAS_DISCRIMINATOR = False
    
    # 创建占位类
    class AdultHolisticDiscriminator:
        def __init__(self, *args, **kwargs):
            raise ImportError("整体判别器未安装")
        
        def score(self, sample):
            return 0.5
        
        def score_batch(self, samples):
            import numpy as np
            return np.ones(len(samples)) * 0.5


def load_discriminator(model_dir: str = "adult_v2/trained_holistic_discriminator"):
    """
    加载已训练的整体判别器
    
    Args:
        model_dir: 模型目录
    
    Returns:
        discriminator: 整体判别器实例
    """
    if not HAS_DISCRIMINATOR:
        print("⚠ 整体判别器不可用")
        return None
    
    import os
    model_path = os.path.join(model_dir, 'holistic_discriminator.pkl')
    
    if not os.path.exists(model_path):
        print(f"⚠ 模型文件不存在: {model_path}")
        print(f"  请先训练模型:")
        print(f"    cd adult_v2")
        print(f"    python -c \"from adult_holistic_discriminator import train_holistic_discriminator; train_holistic_discriminator()\"")
        return None
    
    try:
        discriminator = AdultHolisticDiscriminator()
        discriminator.load_model(model_dir)
        print(f"✓ 整体判别器已加载: {model_dir}")
        return discriminator
    except Exception as e:
        print(f"⚠ 加载模型失败: {e}")
        return None


__all__ = ['AdultHolisticDiscriminator', 'load_discriminator', 'HAS_DISCRIMINATOR']
