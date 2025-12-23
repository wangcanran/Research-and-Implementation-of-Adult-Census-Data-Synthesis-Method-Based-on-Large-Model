"""
清理无用文件脚本
识别并删除临时文件、旧版本代码、重复测试脚本
"""

import os
from pathlib import Path

# 当前目录
base_dir = Path(__file__).parent

print("=" * 80)
print("清理无用文件分析")
print("=" * 80)

# 分类待删除文件
files_to_delete = {
    "临时测试输出CSV": [
        "adult_v2_output_20.csv",
        "holistic_test_output.csv",
        "modular_self_instruct_output.csv",
        "modular_simple_output.csv",
        "output_high_income_educated.csv",
        "output_random.csv",
        "output_senior.csv",
        "output_young_female.csv",
        "quick_test_output.csv",
        "self_instruct_output.csv",
        "test_guided_generation_output.csv",
        "test_synthetic_with_conditional.csv",
    ],
    
    "旧版本/备份代码": [
        "data_generator_gantry_backup.py",
        "adult_data_generator.py",
    ],
    
    "过时的测试脚本": [
        "test_adult_v2.py",
        "test_conditional_distribution.py",
        "test_guided_generation.py",
        "test_holistic_simple.py",
        "test_holistic_vs_field_level.py",
        "test_modular_generator.py",
        "test_self_instruct.py",
        "test_v2_ml_comparison.py",
        "test_v2_simple.py",
        "test_v2_with_discriminative.py",
        "test_all_discriminative_models.py",
    ],
    
    "过时的训练脚本": [
        "train_discriminative_models.py",
        "train_discriminative_models_dl.py",
        "train_discriminative_models_ml.py",
        "train_holistic_discriminator.py",  # 有simple版本
    ],
    
    "临时工具脚本": [
        "check_data_size.py",
        "demo_holistic_discriminator.py",
        "generate_adult_samples.py",
        "adult_config.py",
    ]
}

# 保留的重要文件（明确标注不删除）
important_files = {
    "核心代码": [
        "data_generator.py",  # 原始参考版本，保留
    ],
    "当前使用的测试": [
        "test_with_benchmark.py",
        "test_discriminator_guided.py",
        "test_discriminator_only.py",
    ],
    "当前使用的训练": [
        "train_holistic_discriminator_simple.py",
    ],
    "当前测试输出": [
        "benchmark_test_output.csv",
        "output_discriminator_only.csv",
    ],
    "文档": [
        "WORKFLOW.md",
        "ARCHITECTURE_COMPARISON.md",
        "README_ADULT.md",
    ]
}

# 统计
total_files = 0
total_size = 0

print("\n待删除文件分类：\n")

for category, files in files_to_delete.items():
    print(f"【{category}】")
    category_count = 0
    category_size = 0
    
    for filename in files:
        filepath = base_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            category_count += 1
            category_size += size
            total_files += 1
            total_size += size
            
            size_kb = size / 1024
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size_kb:.1f} KB"
            else:
                size_str = f"{size_kb/1024:.1f} MB"
            
            print(f"  - {filename} ({size_str})")
    
    if category_count > 0:
        print(f"  小计: {category_count}个文件, {category_size/1024:.1f} KB\n")
    else:
        print(f"  (无文件)\n")

print("=" * 80)
print(f"总计待删除: {total_files}个文件, {total_size/1024:.1f} KB")
print("=" * 80)

print("\n保留的重要文件：\n")
for category, files in important_files.items():
    print(f"【{category}】")
    for filename in files:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
    print()

# 询问是否删除
print("=" * 80)
print("是否执行删除？")
print("=" * 80)
response = input("\n输入 'yes' 确认删除，其他任意键取消: ")

if response.lower() == 'yes':
    deleted_count = 0
    deleted_size = 0
    
    print("\n开始删除...")
    
    for category, files in files_to_delete.items():
        for filename in files:
            filepath = base_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                try:
                    filepath.unlink()
                    print(f"  ✓ 已删除: {filename}")
                    deleted_count += 1
                    deleted_size += size
                except Exception as e:
                    print(f"  ✗ 删除失败: {filename} ({e})")
    
    print("\n" + "=" * 80)
    print(f"删除完成！")
    print(f"已删除 {deleted_count} 个文件，释放 {deleted_size/1024:.1f} KB 空间")
    print("=" * 80)
else:
    print("\n取消删除操作")
