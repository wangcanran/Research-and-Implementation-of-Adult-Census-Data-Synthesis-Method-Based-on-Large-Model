#!/usr/bin/env python3
"""对比真实数据与合成数据的关键字段分布"""

import pandas as pd
from collections import Counter

# 读取数据
print("Loading datasets...")
real_data = pd.read_csv("archive/adult.csv")
synthetic_data = pd.read_csv("final_evaluation_3000_samples.csv")

print(f"Real data: {len(real_data)} samples")
print(f"Synthetic data: {len(synthetic_data)} samples")
print("\n" + "="*80)

# 定义关键字段
key_fields = ["age", "education", "occupation", "income"]

# 1. 单字段分布对比
print("\n### SINGLE FIELD DISTRIBUTIONS ###\n")
for field in key_fields:
    print(f"\n{field.upper()}:")
    print("-" * 60)
    
    real_dist = real_data[field].value_counts(normalize=True).head(10)
    synth_dist = synthetic_data[field].value_counts(normalize=True).head(10)
    
    print(f"\nReal Data (Top 10):")
    for val, freq in real_dist.items():
        print(f"  {str(val):30s}: {freq*100:5.1f}%")
    
    print(f"\nSynthetic Data (Top 10):")
    for val, freq in synth_dist.items():
        print(f"  {str(val):30s}: {freq*100:5.1f}%")
    
    # 计算唯一值数量
    real_unique = real_data[field].nunique()
    synth_unique = synthetic_data[field].nunique()
    print(f"\nUnique values: Real={real_unique}, Synthetic={synth_unique}")

# 2. 关键字段组合分析
print("\n\n" + "="*80)
print("### KEY FIELD COMBINATIONS ###\n")

# 生成组合签名
def create_signature(row):
    return f"{row['age']}_{row['education']}_{row['occupation']}_{row['income']}"

real_data['signature'] = real_data.apply(create_signature, axis=1)
synthetic_data['signature'] = synthetic_data.apply(create_signature, axis=1)

# 统计唯一组合
real_unique_combos = real_data['signature'].nunique()
synth_unique_combos = synthetic_data['signature'].nunique()

print(f"Unique combinations in real data: {real_unique_combos}")
print(f"Unique combinations in synthetic data: {synth_unique_combos}")
print(f"Coverage: {synth_unique_combos / real_unique_combos * 100:.1f}%")

# 3. 最常见的组合对比
print("\n\n### TOP 20 MOST FREQUENT COMBINATIONS ###\n")
print("-" * 80)
print(f"{'Rank':<5} {'Age':<5} {'Education':<20} {'Occupation':<25} {'Income':<8} {'Real%':<8} {'Synth%':<8}")
print("-" * 80)

real_combo_counts = real_data['signature'].value_counts()
synth_combo_counts = synthetic_data['signature'].value_counts()

# 获取Top 20合成数据组合
top_synth_combos = synth_combo_counts.head(20)

for rank, (sig, synth_count) in enumerate(top_synth_combos.items(), 1):
    parts = sig.split('_')
    age = parts[0]
    education = parts[1][:18] if len(parts[1]) > 18 else parts[1]
    occupation = parts[2][:23] if len(parts[2]) > 23 else parts[2]
    income = parts[3]
    
    synth_pct = synth_count / len(synthetic_data) * 100
    real_count = real_combo_counts.get(sig, 0)
    real_pct = real_count / len(real_data) * 100
    
    print(f"{rank:<5} {age:<5} {education:<20} {occupation:<25} {income:<8} {real_pct:<8.2f} {synth_pct:<8.2f}")

# 4. 重复率分析
print("\n\n### DUPLICATION ANALYSIS ###\n")
print("-" * 60)

# 真实数据重复率
real_duplicates = len(real_data) - real_unique_combos
real_dup_rate = real_duplicates / len(real_data) * 100

# 合成数据重复率
synth_duplicates = len(synthetic_data) - synth_unique_combos
synth_dup_rate = synth_duplicates / len(synthetic_data) * 100

print(f"Real Data:")
print(f"  Total samples: {len(real_data)}")
print(f"  Unique combinations: {real_unique_combos}")
print(f"  Duplicated samples: {real_duplicates} ({real_dup_rate:.1f}%)")
print(f"  Average copies per combo: {len(real_data) / real_unique_combos:.2f}")

print(f"\nSynthetic Data:")
print(f"  Total samples: {len(synthetic_data)}")
print(f"  Unique combinations: {synth_unique_combos}")
print(f"  Duplicated samples: {synth_duplicates} ({synth_dup_rate:.1f}%)")
print(f"  Average copies per combo: {len(synthetic_data) / synth_unique_combos:.2f}")

# 5. 找出合成数据过度生成的组合
print("\n\n### OVER-GENERATED COMBINATIONS (Top 15) ###\n")
print("-" * 80)
print(f"{'Age':<5} {'Education':<20} {'Occupation':<25} {'Income':<8} {'Real':<8} {'Synth':<8} {'Diff':<8}")
print("-" * 80)

over_gen = []
for sig in synth_combo_counts.head(50).index:
    real_count = real_combo_counts.get(sig, 0)
    synth_count = synth_combo_counts[sig]
    real_pct = real_count / len(real_data) * 100
    synth_pct = synth_count / len(synthetic_data) * 100
    diff = synth_pct - real_pct
    
    if diff > 0.5:  # 差异超过0.5%
        over_gen.append((sig, real_pct, synth_pct, diff))

over_gen.sort(key=lambda x: x[3], reverse=True)

for sig, real_pct, synth_pct, diff in over_gen[:15]:
    parts = sig.split('_')
    age = parts[0]
    education = parts[1][:18] if len(parts[1]) > 18 else parts[1]
    occupation = parts[2][:23] if len(parts[2]) > 23 else parts[2]
    income = parts[3]
    
    print(f"{age:<5} {education:<20} {occupation:<25} {income:<8} {real_pct:<8.2f} {synth_pct:<8.2f} +{diff:<7.2f}")

print("\n" + "="*80)
print("Analysis complete!")
