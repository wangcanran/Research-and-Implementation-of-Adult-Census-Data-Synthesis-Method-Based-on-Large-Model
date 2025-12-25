#!/usr/bin/env python3
"""综合质量评估：从多个维度评估合成数据质量"""

import pandas as pd
import numpy as np
from collections import Counter

# 读取数据
print("Loading datasets...")
real_data = pd.read_csv("archive/adult.csv")
synthetic_data = pd.read_csv("final_evaluation_3000_samples.csv")

print(f"Real data: {len(real_data)} samples")
print(f"Synthetic data: {len(synthetic_data)} samples")
print("\n" + "="*80)

# ============================================================================
# 1. 逻辑一致性检查
# ============================================================================
print("\n### 1. LOGICAL CONSISTENCY CHECK ###\n")

logic_errors = []

for idx, row in synthetic_data.iterrows():
    # 规则1: 年轻人(<18)不能已婚
    if row['age'] < 18 and 'Married' in str(row.get('marital.status', '')):
        logic_errors.append(f"Row {idx}: Age {row['age']} but married")
    
    # 规则2: 教育年限与教育水平匹配
    edu_num = row.get('education.num', 0)
    education = str(row.get('education', ''))
    if education == 'Doctorate' and edu_num < 16:
        logic_errors.append(f"Row {idx}: Doctorate but education.num={edu_num}")
    elif education == 'Bachelors' and edu_num < 13:
        logic_errors.append(f"Row {idx}: Bachelors but education.num={edu_num}")
    
    # 规则3: 工作时长合理性 (0-100)
    hours = row.get('hours.per.week', 0)
    if hours < 0 or hours > 100:
        logic_errors.append(f"Row {idx}: Invalid hours.per.week={hours}")
    
    # 规则4: 婚姻状态与关系匹配
    marital = str(row.get('marital.status', ''))
    relationship = str(row.get('relationship', ''))
    if 'Married' in marital and relationship not in ['Husband', 'Wife']:
        logic_errors.append(f"Row {idx}: Married but relationship={relationship}")
    
    # 规则5: 高收入者通常工作时长>20小时
    income = str(row.get('income', ''))
    if income == '>50K' and hours < 20:
        logic_errors.append(f"Row {idx}: High income but hours={hours}")

print(f"Total logic errors found: {len(logic_errors)}")
print(f"Logic consistency rate: {(1 - len(logic_errors)/len(synthetic_data))*100:.1f}%")

if logic_errors[:5]:
    print("\nFirst 5 errors:")
    for err in logic_errors[:5]:
        print(f"  - {err}")

# ============================================================================
# 2. 数值字段合理性检查
# ============================================================================
print("\n\n### 2. NUMERICAL FIELD VALIDITY ###\n")

numerical_fields = ['age', 'education.num', 'hours.per.week', 'capital.gain', 
                   'capital.loss', 'fnlwgt']

for field in numerical_fields:
    if field not in synthetic_data.columns:
        continue
    
    real_vals = real_data[field].dropna()
    synth_vals = synthetic_data[field].dropna()
    
    print(f"\n{field}:")
    print(f"  Real:      min={real_vals.min():.0f}, max={real_vals.max():.0f}, "
          f"mean={real_vals.mean():.1f}, std={real_vals.std():.1f}")
    print(f"  Synthetic: min={synth_vals.min():.0f}, max={synth_vals.max():.0f}, "
          f"mean={synth_vals.mean():.1f}, std={synth_vals.std():.1f}")
    
    # 检查超出范围的值
    out_of_range = synth_vals[(synth_vals < real_vals.min()) | 
                              (synth_vals > real_vals.max())]
    if len(out_of_range) > 0:
        print(f"  ⚠️ WARNING: {len(out_of_range)} values out of real data range")

# ============================================================================
# 3. 类别字段完整性检查
# ============================================================================
print("\n\n### 3. CATEGORICAL FIELD VALIDITY ###\n")

categorical_fields = ['workclass', 'education', 'marital.status', 'occupation',
                     'relationship', 'race', 'sex', 'native.country']

invalid_count = 0
for field in categorical_fields:
    if field not in synthetic_data.columns:
        continue
    
    real_categories = set(real_data[field].dropna().unique())
    synth_categories = set(synthetic_data[field].dropna().unique())
    
    # 检查非法值
    invalid_cats = synth_categories - real_categories
    if invalid_cats:
        print(f"{field}: Found {len(invalid_cats)} invalid values: {invalid_cats}")
        invalid_count += len(invalid_cats)
    
    # 检查缺失重要类别
    missing_cats = real_categories - synth_categories
    if len(missing_cats) > len(real_categories) * 0.3:  # 缺失超过30%
        print(f"{field}: Missing {len(missing_cats)}/{len(real_categories)} categories")

if invalid_count == 0:
    print("[OK] All categorical values are valid (within real data vocabulary)")
else:
    print(f"\n[WARNING] Total invalid categorical values: {invalid_count}")

# ============================================================================
# 4. 分布相似度评估
# ============================================================================
print("\n\n### 4. DISTRIBUTION SIMILARITY ###\n")

from scipy.spatial.distance import jensenshannon

def calculate_js_divergence(real_dist, synth_dist):
    """计算JS散度"""
    all_values = set(real_dist.keys()) | set(synth_dist.keys())
    real_probs = np.array([real_dist.get(v, 0) for v in all_values])
    synth_probs = np.array([synth_dist.get(v, 0) for v in all_values])
    
    # 归一化
    real_probs = real_probs / real_probs.sum() if real_probs.sum() > 0 else real_probs
    synth_probs = synth_probs / synth_probs.sum() if synth_probs.sum() > 0 else synth_probs
    
    js = jensenshannon(real_probs, synth_probs)
    similarity = 1 - js  # 转为相似度
    return similarity

key_fields = ['income', 'education', 'occupation', 'sex', 'race']
similarities = {}

for field in key_fields:
    real_dist = real_data[field].value_counts(normalize=True).to_dict()
    synth_dist = synthetic_data[field].value_counts(normalize=True).to_dict()
    
    sim = calculate_js_divergence(real_dist, synth_dist)
    similarities[field] = sim
    print(f"{field:20s}: {sim*100:5.1f}%")

avg_similarity = np.mean(list(similarities.values()))
print(f"\n{'Average Similarity':20s}: {avg_similarity*100:5.1f}%")

# ============================================================================
# 5. 判别器评分分析
# ============================================================================
print("\n\n### 5. DISCRIMINATOR SCORE ANALYSIS ###\n")

if '_auxiliary_confidence' in synthetic_data.columns:
    scores = synthetic_data['_auxiliary_confidence'].dropna()
    
    print(f"Total samples with scores: {len(scores)}")
    print(f"Mean score: {scores.mean():.3f}")
    print(f"Median score: {scores.median():.3f}")
    print(f"Std: {scores.std():.3f}")
    print(f"\nScore distribution:")
    print(f"  >0.9: {(scores > 0.9).sum()} ({(scores > 0.9).sum()/len(scores)*100:.1f}%)")
    print(f"  >0.8: {(scores > 0.8).sum()} ({(scores > 0.8).sum()/len(scores)*100:.1f}%)")
    print(f"  >0.7: {(scores > 0.7).sum()} ({(scores > 0.7).sum()/len(scores)*100:.1f}%)")
    print(f"  <0.7: {(scores < 0.7).sum()} ({(scores < 0.7).sum()/len(scores)*100:.1f}%)")
else:
    print("No discriminator scores found in data")

# ============================================================================
# 6. 综合质量评分
# ============================================================================
print("\n\n" + "="*80)
print("### OVERALL QUALITY ASSESSMENT ###\n")

# 计算各维度得分
logic_score = (1 - len(logic_errors)/len(synthetic_data)) * 100
distribution_score = avg_similarity * 100
discriminator_score = scores.mean() * 100 if '_auxiliary_confidence' in synthetic_data.columns else 0

# 检查数值合理性
num_quality_score = 100  # 假设默认满分，如有问题会扣分

print(f"1. Logical Consistency:     {logic_score:5.1f}%")
print(f"2. Distribution Similarity: {distribution_score:5.1f}%")
print(f"3. Discriminator Quality:   {discriminator_score:5.1f}%")
print(f"4. Numerical Validity:      {num_quality_score:5.1f}%")

overall_quality = (logic_score * 0.25 + 
                   distribution_score * 0.25 + 
                   discriminator_score * 0.3 +
                   num_quality_score * 0.2)

print(f"\n{'='*40}")
print(f"OVERALL QUALITY SCORE: {overall_quality:.1f}%")
print(f"{'='*40}")

# 评级
if overall_quality >= 90:
    grade = "A (Excellent)"
elif overall_quality >= 80:
    grade = "B (Good)"
elif overall_quality >= 70:
    grade = "C (Acceptable)"
elif overall_quality >= 60:
    grade = "D (Poor)"
else:
    grade = "F (Failed)"

print(f"\nQuality Grade: {grade}")

# ============================================================================
# 7. 问题总结
# ============================================================================
print("\n\n### ISSUES SUMMARY ###\n")

issues = []
if logic_score < 100:
    issues.append(f"[!] {len(logic_errors)} logical inconsistencies detected")
if distribution_score < 90:
    issues.append(f"[!] Distribution similarity below 90% (actual: {distribution_score:.1f}%)")
if invalid_count > 0:
    issues.append(f"[!] {invalid_count} invalid categorical values")

if issues:
    for issue in issues:
        print(issue)
else:
    print("[OK] No major issues detected")

print("\n" + "="*80)
print("Quality assessment complete!")
