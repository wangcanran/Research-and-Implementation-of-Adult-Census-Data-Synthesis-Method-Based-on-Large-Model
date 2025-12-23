"""
简单示例脚本：使用 AdultDataGenerator 生成合成数据
"""
from adult_data_generator import AdultDataGenerator, GenerationCondition

def main():
    print("=" * 70)
    print("Adult Census 合成数据生成示例")
    print("=" * 70)
    
    # 初始化生成器
    print("\n[步骤1] 初始化生成器...")
    generator = AdultDataGenerator(
        sample_file="archive/adult.csv",  # 使用真实数据学习统计特征
        use_heuristic=True,               # 启用启发式示例选择
        verbose=True                      # 显示详细信息
    )
    
    # 场景1: 生成高收入、高教育、中年人群
    print("\n" + "=" * 70)
    print("[场景1] 高收入、高学历、中年人群 (30个样本)")
    print("=" * 70)
    condition_1 = GenerationCondition(
        age_range="middle",           # 31-55岁
        education_level="high",       # 高学历 (Bachelors+)
        income_class=">50K",          # 高收入
        gender=None,                  # 性别不限
        marital_status=None           # 婚姻状况不限
    )
    samples_1 = generator.generate_batch(30, condition_1)
    generator.save_to_csv(samples_1, "output_high_income_educated.csv")
    
    # 场景2: 生成低收入、年轻人群
    print("\n" + "=" * 70)
    print("[场景2] 低收入、年轻人群 (30个样本)")
    print("=" * 70)
    condition_2 = GenerationCondition(
        age_range="young",            # 17-30岁
        education_level="medium",     # 中等学历
        income_class="<=50K",         # 低收入
        gender="Female",              # 女性
        marital_status=None
    )
    samples_2 = generator.generate_batch(30, condition_2)
    generator.save_to_csv(samples_2, "output_young_female.csv")
    
    # 场景3: 生成老年人群
    print("\n" + "=" * 70)
    print("[场景3] 老年人群 (30个样本)")
    print("=" * 70)
    condition_3 = GenerationCondition(
        age_range="senior",           # 56-90岁
        education_level=None,         # 教育水平不限
        income_class=None,            # 收入不限
        gender=None,
        marital_status=None
    )
    samples_3 = generator.generate_batch(30, condition_3)
    generator.save_to_csv(samples_3, "output_senior.csv")
    
    # 场景4: 完全随机生成（无条件约束）
    print("\n" + "=" * 70)
    print("[场景4] 无条件随机生成 (50个样本)")
    print("=" * 70)
    samples_4 = generator.generate_batch(50)
    generator.save_to_csv(samples_4, "output_random.csv")
    
    # 统计信息
    print("\n" + "=" * 70)
    print("生成统计总结")
    print("=" * 70)
    print(f"总计生成样本数: {len(samples_1) + len(samples_2) + len(samples_3) + len(samples_4)}")
    print(f"  - 场景1 (高收入高学历): {len(samples_1)} 个")
    print(f"  - 场景2 (年轻女性): {len(samples_2)} 个")
    print(f"  - 场景3 (老年人群): {len(samples_3)} 个")
    print(f"  - 场景4 (随机): {len(samples_4)} 个")
    print("\n输出文件:")
    print("  - output_high_income_educated.csv")
    print("  - output_young_female.csv")
    print("  - output_senior.csv")
    print("  - output_random.csv")
    print("=" * 70)

if __name__ == "__main__":
    main()
