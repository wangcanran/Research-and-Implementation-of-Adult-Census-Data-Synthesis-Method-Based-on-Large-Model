"""
测试引导式生成策略 - 验证LLM在条件分布约束下的生成效果
"""
from adult_data_generator import AdultDataGenerator, GenerationCondition
import pandas as pd

print("=" * 80)
print("引导式生成策略测试")
print("=" * 80)

# 初始化生成器
print("\n[1] 初始化生成器（加载条件分布）...")
generator = AdultDataGenerator(
    sample_file="archive/adult.csv",
    use_heuristic=True,
    verbose=True
)

# 生成测试样本
print("\n[2] 生成20个测试样本...")
samples = generator.generate_batch(20)

print(f"\n成功生成: {len(samples)} 条")

if samples:
    df = pd.DataFrame(samples)
    
    # 显示生成的字段
    print(f"\n生成的字段: {list(df.columns)}")
    
    # 统计分析
    print("\n" + "=" * 80)
    print("数据质量分析")
    print("=" * 80)
    
    print(f"\n【多样性】")
    print(f"年龄范围: {df['age'].min()}-{df['age'].max()} 岁")
    print(f"性别分布: Male={len(df[df['sex']=='Male'])}, Female={len(df[df['sex']=='Female'])}")
    print(f"种族分布:\n{df['race'].value_counts()}")
    
    print(f"\n【教育程度】")
    print(df['education'].value_counts().head())
    
    print(f"\n【职业分布】")
    print(df['occupation'].value_counts().head())
    
    print(f"\n【婚姻状况】")
    print(df['marital.status'].value_counts())
    
    print(f"\n【收入分布】")
    income_dist = df['income'].value_counts()
    print(f"<=50K: {income_dist.get('<=50K', 0)} ({income_dist.get('<=50K', 0)/len(df):.1%})")
    print(f">50K: {income_dist.get('>50K', 0)} ({income_dist.get('>50K', 0)/len(df):.1%})")
    
    # 检查关键组合
    print(f"\n【关键组合检查】")
    
    # 检查之前的"典型组合"是否还过度出现
    typical = df[
        (df['sex'] == 'Female') & 
        (df['race'] == 'White') & 
        (df['education'] == 'Bachelors') &
        (df['occupation'] == 'Exec-managerial') &
        (df['marital.status'] == 'Married-civ-spouse') &
        (df['income'] == '>50K')
    ]
    print(f"Female+White+Bachelors+Exec-managerial+Married+>50K: {len(typical)}/{len(df)} ({len(typical)/len(df):.1%})")
    print(f"  预期: ~0.14% | 实际: {len(typical)/len(df):.1%}")
    
    # 显示前10条数据
    print("\n" + "=" * 80)
    print("前10条样本")
    print("=" * 80)
    print(df[['age', 'sex', 'education', 'occupation', 'marital.status', 'income']].head(10).to_string())
    
    # 保存
    df.to_csv("test_guided_generation_output.csv", index=False)
    print(f"\n已保存到: test_guided_generation_output.csv")
    
else:
    print("\n⚠ 未生成任何样本")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
