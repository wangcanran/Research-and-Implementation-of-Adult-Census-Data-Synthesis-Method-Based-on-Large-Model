"""
测试条件分布学习效果
对比原始数据和生成数据的条件分布差异
"""
from adult_data_generator import AdultDataGenerator, GenerationCondition
import pandas as pd
import numpy as np

def analyze_conditional_distributions(original_df, synthetic_df):
    """分析条件分布质量"""
    
    print("=" * 80)
    print("条件分布质量分析")
    print("=" * 80)
    
    # 检查必需字段是否存在
    required_fields = ['education', 'income', 'education.num', 'occupation', 'hours.per.week', 
                      'age', 'marital.status', 'sex', 'relationship', 'capital.gain']
    missing_fields = [f for f in required_fields if f not in synthetic_df.columns]
    
    if missing_fields:
        print(f"\n⚠ 错误：生成的数据缺少以下字段: {missing_fields}")
        print(f"实际生成的字段: {list(synthetic_df.columns)}")
        print("\n跳过条件分布分析。")
        return
    
    # 1. P(income|education) - 教育→收入
    print("\n【1】P(income|education) - 教育程度对收入的影响")
    print("-" * 80)
    
    for edu in ['HS-grad', 'Bachelors', 'Masters', 'Doctorate']:
        if edu in original_df['education'].values:
            orig_high = (original_df[original_df['education'] == edu]['income'] == '>50K').mean()
            
            if edu in synthetic_df['education'].values:
                syn_high = (synthetic_df[synthetic_df['education'] == edu]['income'] == '>50K').mean()
                diff = abs(orig_high - syn_high)
                
                status = "✓" if diff < 0.1 else ("⚠" if diff < 0.2 else "✗")
                print(f"{status} {edu:15s} -> >50K概率: 原始={orig_high:.2%}, 生成={syn_high:.2%}, 差异={diff:.2%}")
            else:
                print(f"  {edu:15s} -> 生成数据中未出现")
    
    # 2. P(occupation|education) - 教育→职业
    print("\n【2】P(occupation|education) - 教育程度对职业的影响")
    print("-" * 80)
    
    for edu_level, edu_range in [('低学历', (1, 8)), ('中学历', (9, 12)), ('高学历', (13, 16))]:
        orig_edu = original_df[(original_df['education.num'] >= edu_range[0]) & 
                               (original_df['education.num'] <= edu_range[1])]
        syn_edu = synthetic_df[(synthetic_df['education.num'] >= edu_range[0]) & 
                               (synthetic_df['education.num'] <= edu_range[1])]
        
        if len(orig_edu) > 0 and len(syn_edu) > 0:
            orig_top_occ = orig_edu['occupation'].value_counts().head(3).index.tolist()
            syn_top_occ = syn_edu['occupation'].value_counts().head(3).index.tolist()
            
            overlap = len(set(orig_top_occ) & set(syn_top_occ))
            status = "✓" if overlap >= 2 else ("⚠" if overlap >= 1 else "✗")
            
            print(f"{status} {edu_level} Top3职业重合度: {overlap}/3")
            print(f"    原始: {orig_top_occ}")
            print(f"    生成: {syn_top_occ}")
    
    # 3. P(hours|education) - 教育→工作时长
    print("\n【3】P(hours|education) - 教育程度对工作时长的影响")
    print("-" * 80)
    
    for edu_level, edu_range in [('低学历', (1, 8)), ('中学历', (9, 12)), ('高学历', (13, 16))]:
        orig_edu = original_df[(original_df['education.num'] >= edu_range[0]) & 
                               (original_df['education.num'] <= edu_range[1])]
        syn_edu = synthetic_df[(synthetic_df['education.num'] >= edu_range[0]) & 
                               (synthetic_df['education.num'] <= edu_range[1])]
        
        if len(orig_edu) > 0 and len(syn_edu) > 0:
            orig_mean = orig_edu['hours.per.week'].mean()
            syn_mean = syn_edu['hours.per.week'].mean()
            diff = abs(orig_mean - syn_mean)
            
            status = "✓" if diff < 3 else ("⚠" if diff < 5 else "✗")
            print(f"{status} {edu_level} 平均工时: 原始={orig_mean:.1f}h, 生成={syn_mean:.1f}h, 差异={diff:.1f}h")
    
    # 4. P(marital|age) - 年龄→婚姻状况
    print("\n【4】P(marital|age) - 年龄对婚姻状况的影响")
    print("-" * 80)
    
    for age_group, age_range in [('年轻', (17, 30)), ('中年', (31, 55)), ('老年', (56, 90))]:
        orig_age = original_df[(original_df['age'] >= age_range[0]) & 
                               (original_df['age'] <= age_range[1])]
        syn_age = synthetic_df[(synthetic_df['age'] >= age_range[0]) & 
                               (synthetic_df['age'] <= age_range[1])]
        
        if len(orig_age) > 0 and len(syn_age) > 0:
            orig_married = (orig_age['marital.status'] == 'Married-civ-spouse').mean()
            syn_married = (syn_age['marital.status'] == 'Married-civ-spouse').mean()
            diff = abs(orig_married - syn_married)
            
            status = "✓" if diff < 0.1 else ("⚠" if diff < 0.2 else "✗")
            print(f"{status} {age_group}人 已婚比例: 原始={orig_married:.2%}, 生成={syn_married:.2%}, 差异={diff:.2%}")
    
    # 5. P(relationship|marital,sex) - 婚姻+性别→家庭关系
    print("\n【5】P(relationship|marital,sex) - 婚姻状况+性别对家庭关系的影响")
    print("-" * 80)
    
    for marital in ['Married-civ-spouse', 'Never-married']:
        for sex in ['Male', 'Female']:
            orig_subset = original_df[(original_df['marital.status'] == marital) & 
                                     (original_df['sex'] == sex)]
            syn_subset = synthetic_df[(synthetic_df['marital.status'] == marital) & 
                                      (synthetic_df['sex'] == sex)]
            
            if len(orig_subset) > 10 and len(syn_subset) > 10:
                orig_rel = orig_subset['relationship'].mode()[0] if len(orig_subset['relationship'].mode()) > 0 else 'N/A'
                syn_rel = syn_subset['relationship'].mode()[0] if len(syn_subset['relationship'].mode()) > 0 else 'N/A'
                
                status = "✓" if orig_rel == syn_rel else "✗"
                print(f"{status} {marital:20s} + {sex:6s} -> 主要关系: 原始={orig_rel:15s}, 生成={syn_rel:15s}")
    
    # 6. P(capital.gain|education) - 教育→资本收益
    print("\n【6】P(capital.gain|education) - 教育程度对资本收益的影响")
    print("-" * 80)
    
    for edu_level, edu_range in [('低学历', (1, 8)), ('中学历', (9, 12)), ('高学历', (13, 16))]:
        orig_edu = original_df[(original_df['education.num'] >= edu_range[0]) & 
                               (original_df['education.num'] <= edu_range[1])]
        syn_edu = synthetic_df[(synthetic_df['education.num'] >= edu_range[0]) & 
                               (synthetic_df['education.num'] <= edu_range[1])]
        
        if len(orig_edu) > 0 and len(syn_edu) > 0:
            orig_has_gain = (orig_edu['capital.gain'] > 0).mean()
            syn_has_gain = (syn_edu['capital.gain'] > 0).mean()
            diff = abs(orig_has_gain - syn_has_gain)
            
            status = "✓" if diff < 0.1 else ("⚠" if diff < 0.15 else "✗")
            print(f"{status} {edu_level} 有资本收益比例: 原始={orig_has_gain:.2%}, 生成={syn_has_gain:.2%}, 差异={diff:.2%}")
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


def main():
    print("=" * 80)
    print("条件分布学习效果测试")
    print("=" * 80)
    
    # 加载原始数据
    print("\n[步骤1] 加载原始数据...")
    original_df = pd.read_csv("archive/adult.csv")
    print(f"  原始数据: {len(original_df)} 条")
    
    # 初始化生成器（带条件分布学习）
    print("\n[步骤2] 初始化生成器（启用条件分布学习）...")
    generator = AdultDataGenerator(
        sample_file="archive/adult.csv",
        use_heuristic=True,
        verbose=True
    )
    
    # 生成合成数据
    print("\n[步骤3] 生成合成数据（500个样本）...")
    samples = generator.generate_batch(500)
    synthetic_df = pd.DataFrame(samples)
    print(f"  成功生成: {len(synthetic_df)} 条")
    
    # 调试：打印生成的字段
    print(f"\n  生成的字段: {list(synthetic_df.columns)}")
    print(f"  缺失的字段: {set(original_df.columns) - set(synthetic_df.columns)}")
    
    # 检查是否有空样本
    if len(synthetic_df) == 0:
        print("  ⚠ 警告：未生成任何样本！")
        return
    
    # 保存生成数据
    synthetic_df.to_csv("test_synthetic_with_conditional.csv", index=False)
    print("  已保存到: test_synthetic_with_conditional.csv")
    
    # 分析条件分布质量
    print("\n[步骤4] 分析条件分布质量...")
    analyze_conditional_distributions(original_df, synthetic_df)
    
    # 生成统计对比报告
    print("\n" + "=" * 80)
    print("边缘分布对比")
    print("=" * 80)
    
    print("\n年龄分布:")
    print(f"  原始: 均值={original_df['age'].mean():.1f}, 标准差={original_df['age'].std():.1f}")
    print(f"  生成: 均值={synthetic_df['age'].mean():.1f}, 标准差={synthetic_df['age'].std():.1f}")
    
    print("\n工作时长分布:")
    print(f"  原始: 均值={original_df['hours.per.week'].mean():.1f}, 标准差={original_df['hours.per.week'].std():.1f}")
    print(f"  生成: 均值={synthetic_df['hours.per.week'].mean():.1f}, 标准差={synthetic_df['hours.per.week'].std():.1f}")
    
    print("\n收入分布:")
    print(f"  原始: >50K={((original_df['income'] == '>50K').mean()):.2%}")
    print(f"  生成: >50K={((synthetic_df['income'] == '>50K').mean()):.2%}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
