"""
Adult Data Generator V2 - 简单测试
完整三阶段流程：Generation → Curation → Evaluation
生成20条数据
"""

import sys
sys.path.append('adult_v2')

from adult_v2.adult_generator_main import AdultDataGenerator

print("=" * 80)
print("Adult Data Generator V2 - 完整流程测试（20条数据）")
print("=" * 80)

# 初始化生成器
print("\n[1] 初始化生成器...")
generator = AdultDataGenerator(use_advanced_features=True)

# 加载真实数据
print("\n[2] 加载真实数据并学习统计特征...")
generator.load_real_samples("archive/adult.csv", limit=1000)

# 完整三阶段流程
print("\n[3] 运行完整三阶段流程（Generation + Curation + Evaluation）...")
result = generator.generate_with_full_pipeline(n_samples=20, use_scheduler=True)

# 结果统计
print("\n" + "=" * 80)
print("结果统计")
print("=" * 80)

print(f"\n生成样本: {len(result['generated_samples'])}")
print(f"策展后样本: {len(result['curated_samples'])}")

# 评估结果
eval_results = result['evaluation_results']
print(f"\n评估结果：")
print(f"  格式正确率: {eval_results['format_correctness']['validity_rate']:.1%}")
print(f"  逻辑一致率: {eval_results['logical_consistency']['consistency_rate']:.1%}")
if 'faithfulness' in eval_results:
    print(f"  分布相似度: {eval_results['faithfulness']['overall_similarity']:.1%}")
print(f"  总体评分: {eval_results['overall_score']:.1%}")

# 保存结果
print("\n[4] 保存结果...")
generator.save_to_csv(result['curated_samples'], "adult_v2_output_20.csv")

# 显示前5条样本
print("\n" + "=" * 80)
print("前5条样本预览")
print("=" * 80)

import pandas as pd
df = pd.DataFrame(result['curated_samples'])
print("\n关键字段:")
key_fields = ['age', 'sex', 'education', 'occupation', 'marital.status', 'hours.per.week', 'income']
if all(f in df.columns for f in key_fields):
    print(df[key_fields].head().to_string())

# 数据质量分析
print("\n" + "=" * 80)
print("数据质量分析")
print("=" * 80)

print(f"\n年龄分布: {df['age'].min()}-{df['age'].max()} 岁 (均值: {df['age'].mean():.1f})")
print(f"\n性别分布:")
print(df['sex'].value_counts())

print(f"\n收入分布:")
income_dist = df['income'].value_counts()
for income, count in income_dist.items():
    print(f"  {income}: {count} ({count/len(df):.1%})")

print(f"\n教育程度（Top 3）:")
print(df['education'].value_counts().head(3))

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
print(f"\n生成文件: adult_v2_output_20.csv")
print(f"样本数量: {len(result['curated_samples'])}")
print(f"总体评分: {eval_results['overall_score']:.1%}")
