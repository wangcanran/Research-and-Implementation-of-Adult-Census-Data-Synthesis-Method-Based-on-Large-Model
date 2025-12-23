"""
直接删除无用文件
"""

import os
from pathlib import Path

base_dir = Path(__file__).parent

files_to_delete = [
    # Temporary CSV outputs
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
    
    # Old/backup code
    "data_generator_gantry_backup.py",
    "adult_data_generator.py",
    
    # Old test scripts
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
    
    # Old training scripts
    "train_discriminative_models.py",
    "train_discriminative_models_dl.py",
    "train_discriminative_models_ml.py",
    "train_holistic_discriminator.py",
    
    # Temporary utility scripts
    "check_data_size.py",
    "demo_holistic_discriminator.py",
    "generate_adult_samples.py",
    "adult_config.py",
]

print("Deleting unused files...")
print("=" * 80)

deleted_count = 0
deleted_size = 0
not_found = []

for filename in files_to_delete:
    filepath = base_dir / filename
    if filepath.exists():
        size = filepath.stat().st_size
        try:
            filepath.unlink()
            print(f"Deleted: {filename} ({size} bytes)")
            deleted_count += 1
            deleted_size += size
        except Exception as e:
            print(f"Failed: {filename} ({e})")
    else:
        not_found.append(filename)

print("=" * 80)
print(f"Total: Deleted {deleted_count} files, freed {deleted_size/1024:.1f} KB")

if not_found:
    print(f"\nNot found: {len(not_found)} files")

print("\nDone!")
