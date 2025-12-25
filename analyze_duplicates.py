"""
分析3000条生成样本的重复情况
"""

import pandas as pd
from collections import Counter

# 加载数据
df = pd.read_csv("final_evaluation_3000_samples.csv")

print("=" * 80)
print("样本重复分析")
print("=" * 80)

print(f"\n总样本数: {len(df)}")

# 1. 完全重复的样本
duplicate_rows = df.duplicated(keep=False)
print(f"\n[1] 完全重复的样本:")
print(f"  重复样本数: {duplicate_rows.sum()}")
print(f"  独特样本数: {(~df.duplicated()).sum()}")
print(f"  重复率: {duplicate_rows.sum()/len(df):.1%}")

# 2. 按所有字段分组统计
all_columns = df.columns.tolist()
grouped = df.groupby(all_columns).size().reset_index(name='count')
grouped_sorted = grouped.sort_values('count', ascending=False)

print(f"\n[2] 出现次数最多的样本（Top 10）:")
for i, row in grouped_sorted.head(10).iterrows():
    count = row['count']
    print(f"\n  出现 {count} 次:")
    sample_dict = row.drop('count').to_dict()
    for k, v in list(sample_dict.items())[:5]:  # 只显示前5个字段
        print(f"    {k}: {v}")

# 3. 关键字段组合的重复
key_fields = ['age', 'education', 'occupation', 'marital.status', 'sex', 'income']
if all(f in df.columns for f in key_fields):
    key_duplicates = df[key_fields].duplicated(keep=False)
    print(f"\n[3] 关键字段组合重复:")
    print(f"  重复样本数: {key_duplicates.sum()}")
    print(f"  重复率: {key_duplicates.sum()/len(df):.1%}")

# 4. 各维度的唯一值数量
print(f"\n[4] 各字段唯一值统计:")
for col in ['income', 'sex', 'race', 'education', 'occupation', 'marital.status']:
    if col in df.columns:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} 种")

print(f"\n  age: {df['age'].nunique()} 种 (范围: {df['age'].min()}-{df['age'].max()})")

# 5. 收入分布
if 'income' in df.columns:
    print(f"\n[5] 收入分布:")
    income_dist = df['income'].value_counts()
    for income, count in income_dist.items():
        print(f"  {income}: {count} ({count/len(df):.1%})")

# 6. 年龄段分布
if 'age' in df.columns:
    print(f"\n[6] 年龄段分布:")
    df_copy = df.copy()
    df_copy['age_range'] = df_copy['age'].apply(
        lambda x: 'young' if x <= 30 else ('senior' if x > 55 else 'middle')
    )
    age_dist = df_copy['age_range'].value_counts()
    for age_range, count in age_dist.items():
        print(f"  {age_range}: {count} ({count/len(df):.1%})")

# 7. 教育水平分布
if 'education.num' in df.columns:
    print(f"\n[7] 教育水平分布:")
    df_copy['edu_level'] = df_copy['education.num'].apply(
        lambda x: 'low' if x <= 8 else ('high' if x > 12 else 'medium')
    )
    edu_dist = df_copy['edu_level'].value_counts()
    for edu, count in edu_dist.items():
        print(f"  {edu}: {count} ({count/len(df):.1%})")

print("\n" + "=" * 80)
