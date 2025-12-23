"""
Adult Data Generator V2 - 测试脚本
基于论文三阶段框架的完整测试
"""

import sys
sys.path.append('adult_v2')

from adult_v2.adult_generator_main import AdultDataGenerator
from adult_v2.adult_task_spec import GenerationCondition

print("=" * 80)
print("Adult Data Generator V2 - 完整测试")
print("基于论文三阶段框架：Generation → Curation → Evaluation")
print("=" * 80)

# ============================================================================
#                      测试1：基础生成（仅Generation）
# ============================================================================

print("\n" + "=" * 80)
print("测试1：基础生成（50个样本）")
print("=" * 80)

# 初始化生成器
generator = AdultDataGenerator(use_advanced_features=True)

# 加载真实数据
print("\n加载真实数据...")
generator.load_real_samples("archive/adult.csv", limit=1000)

# 生成样本
print("\n开始生成...")
samples = generator.generate_batch(n_samples=50, use_scheduler=True)

print(f"\n生成结果：{len(samples)} 个样本")

# 保存
generator.save_to_csv(samples, "test_v2_basic_output.csv")

# ============================================================================
#                      测试2：生成+策展（Generation + Curation）
# ============================================================================

print("\n" + "=" * 80)
print("测试2：生成+策展（50个样本）")
print("=" * 80)

result = generator.generate_with_curation(n_samples=50, use_scheduler=True)

print(f"\n策展结果：")
print(f"  原始样本: {len(result['generated_samples'])}")
print(f"  策展后样本: {len(result['curated_samples'])}")
print(f"  过滤统计: {result['curation_stats']['filter_stats']}")

# 保存策展后的样本
generator.save_to_csv(result['curated_samples'], "test_v2_curated_output.csv")

# ============================================================================
#                      测试3：完整流程（Generation + Curation + Evaluation）
# ============================================================================

print("\n" + "=" * 80)
print("测试3：完整三阶段流程（100个样本）")
print("=" * 80)

full_result = generator.generate_with_full_pipeline(n_samples=100, use_scheduler=True)

print(f"\n完整流程结果：")
print(f"  生成样本: {len(full_result['generated_samples'])}")
print(f"  策展后样本: {len(full_result['curated_samples'])}")

# 评估结果
eval_results = full_result['evaluation_results']
print(f"\n评估结果：")
print(f"  格式正确率: {eval_results['format_correctness']['validity_rate']:.1%}")
print(f"  逻辑一致率: {eval_results['logical_consistency']['consistency_rate']:.1%}")
if 'faithfulness' in eval_results:
    print(f"  分布相似度: {eval_results['faithfulness']['overall_similarity']:.1%}")
print(f"  总体评分: {eval_results['overall_score']:.1%}")

# 保存最终样本
generator.save_to_csv(full_result['curated_samples'], "test_v2_full_pipeline_output.csv")

# ============================================================================
#                      测试4：条件生成
# ============================================================================

print("\n" + "=" * 80)
print("测试4：条件生成（指定条件）")
print("=" * 80)

# 生成高收入、高学历的中年人
condition = GenerationCondition(
    age_range="middle",
    education_level="high",
    income_class=">50K",
    gender="Male"
)

print(f"\n生成条件: {condition}")

conditional_samples = generator.generate_batch(
    n_samples=20,
    condition=condition,
    use_scheduler=False  # 不使用调度器，严格按条件生成
)

print(f"\n生成了 {len(conditional_samples)} 个样本")

# 验证条件
high_edu_count = sum(1 for s in conditional_samples if s.get('education.num', 0) >= 13)
high_income_count = sum(1 for s in conditional_samples if s.get('income') == '>50K')
male_count = sum(1 for s in conditional_samples if s.get('sex') == 'Male')

print(f"\n条件验证：")
print(f"  高学历(>=13): {high_edu_count}/{len(conditional_samples)} ({high_edu_count/len(conditional_samples):.1%})")
print(f"  高收入(>50K): {high_income_count}/{len(conditional_samples)} ({high_income_count/len(conditional_samples):.1%})")
print(f"  男性: {male_count}/{len(conditional_samples)} ({male_count/len(conditional_samples):.1%})")

generator.save_to_csv(conditional_samples, "test_v2_conditional_output.csv")

# ============================================================================
#                      测试5：调度器分布统计
# ============================================================================

print("\n" + "=" * 80)
print("测试5：调度器分布统计")
print("=" * 80)

scheduler_stats = generator.get_scheduler_stats()

print("\n当前分布 vs 目标分布：")
for dimension, current_dist in scheduler_stats['current_distribution'].items():
    target_dist = scheduler_stats['target_distribution'].get(dimension, {})
    print(f"\n  {dimension}:")
    for category in target_dist.keys():
        current = current_dist.get(category, 0)
        target = target_dist.get(category, 0)
        print(f"    {category}: {current:.2%} (目标: {target:.2%})")

# ============================================================================
#                      测试6：数据质量分析
# ============================================================================

print("\n" + "=" * 80)
print("测试6：数据质量详细分析")
print("=" * 80)

import pandas as pd

# 加载生成的数据
df = pd.DataFrame(full_result['curated_samples'])

print(f"\n数据维度: {df.shape}")
print(f"\n字段列表: {list(df.columns)}")

print(f"\n多样性分析：")
print(f"  年龄范围: {df['age'].min()}-{df['age'].max()} 岁")
print(f"  年龄均值: {df['age'].mean():.1f} ± {df['age'].std():.1f}")

print(f"\n性别分布:")
print(df['sex'].value_counts())

print(f"\n教育程度分布（Top 5）:")
print(df['education'].value_counts().head())

print(f"\n职业分布（Top 5）:")
print(df['occupation'].value_counts().head())

print(f"\n收入分布:")
income_dist = df['income'].value_counts()
for income, count in income_dist.items():
    print(f"  {income}: {count} ({count/len(df):.1%})")

print(f"\n工作时长统计:")
print(f"  均值: {df['hours.per.week'].mean():.1f} ± {df['hours.per.week'].std():.1f}")
print(f"  范围: {df['hours.per.week'].min()}-{df['hours.per.week'].max()}")

# 检查关键组合
print(f"\n关键组合检查（教育-收入）:")
edu_income = df.groupby(['education', 'income']).size().sort_values(ascending=False)
print(edu_income.head(10))

# ============================================================================
#                      总结
# ============================================================================

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

print("\n生成的文件：")
print("  1. test_v2_basic_output.csv - 基础生成（50样本）")
print("  2. test_v2_curated_output.csv - 策展后（50样本）")
print("  3. test_v2_full_pipeline_output.csv - 完整流程（100样本）")
print("  4. test_v2_conditional_output.csv - 条件生成（20样本）")

print("\n关键指标：")
print(f"  格式正确率: {eval_results['format_correctness']['validity_rate']:.1%}")
print(f"  逻辑一致率: {eval_results['logical_consistency']['consistency_rate']:.1%}")
print(f"  总体评分: {eval_results['overall_score']:.1%}")

print("\nV2版本特性：")
print("  ✓ 完整三阶段框架（Generation + Curation + Evaluation）")
print("  ✓ 启发式示例选择（DemonstrationManager）")
print("  ✓ 目标分布调度（DatasetWiseScheduler）")
print("  ✓ 条件分布学习（9种条件分布）")
print("  ✓ 引导式Prompt生成")
print("  ✓ 样本过滤和重加权")
print("  ✓ 多维度评估体系")

print("\n" + "=" * 80)
