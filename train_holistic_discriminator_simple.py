"""
训练整体判别器 - 简化版本
"""

import sys
import os

# 添加adult_v2到路径
sys.path.insert(0, 'adult_v2')

from adult_holistic_discriminator import train_holistic_discriminator

print("=" * 80)
print("训练整体判别器（Holistic Discriminator）")
print("=" * 80)

print("\nCore Idea:")
print("  - Learn to distinguish real vs generated samples")
print("  - Automatically learn distribution features")
print("  - No need to manually define causal relationships")
print("  - Avoid circular dependencies")

print("\nAdvantages:")
print("  1. No preset field relationships")
print("  2. Model discovers patterns itself")
print("  3. Scores each sample [0,1] for authenticity")
print("  4. Can filter low-quality samples")

print("\n训练策略：")
print("  正样本：真实Adult数据")
print("  负样本：字段打乱的假数据（破坏真实分布）")

print("\n" + "=" * 80)

# 训练
discriminator, result = train_holistic_discriminator(
    data_file="archive/adult.csv",
    model_dir="adult_v2/trained_holistic_discriminator",
    model_type='gradient_boosting',  # 或 'random_forest'
    sample_limit=10000  # 使用1万条数据训练
)

print("\n" + "=" * 80)
print("训练完成！")
print("=" * 80)

print(f"\n模型性能：")
print(f"  准确率: {result['accuracy']:.1%}")
print(f"  AUC: {result['auc']:.3f}")

print("\n模型已保存到: adult_v2/trained_holistic_discriminator/")
print("\n现在可以运行:")
print("  python test_discriminator_guided.py")
