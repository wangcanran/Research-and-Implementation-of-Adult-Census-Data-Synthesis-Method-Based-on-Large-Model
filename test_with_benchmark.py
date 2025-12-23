"""
Test with Benchmark Evaluation
测试完整的三阶段框架：Generation → Curation → Evaluation
"""

import sys
sys.path.append('generator_modules')

from generator_modules.main_generator import DataGenerator
import pandas as pd

print("=" * 80)
print("Complete Framework Test - Generation → Curation → Evaluation")
print("=" * 80)

print("\nFramework (from data_generator.py):")
print("  I.   Generation: Sample-Wise Decomposition + Dataset-Wise Scheduling")
print("  II.  Curation: Filter + Auxiliary Model (Discriminator)")
print("  III. Evaluation: Direct + Benchmark (vs real data)")

print("\n" + "=" * 80)
print("Test: Self-Instruct with Benchmark Evaluation")
print("=" * 80)

# 初始化生成器
generator = DataGenerator(verbose=True)

# 加载真实数据（会自动加载到Benchmark评估器）
generator.load_real_samples("archive/adult.csv", limit=1000)

# 使用Self-Instruct生成
print("\n[Self-Instruct] Starting iterative generation...")
seed_samples = generator.demo_manager.real_samples[:10]

generated_samples = generator.iterative_generate(
    seed_samples=seed_samples,
    target_count=100,
    n_iterations=3,
    quality_threshold=0.6
)

print(f"\n[Result] Generated {len(generated_samples)} high-quality samples")

# Benchmark评估（与真实数据对比）
print("\n" + "=" * 80)
print("Benchmark Evaluation (vs Real Data)")
print("=" * 80)

benchmark_results = generator.benchmark_evaluator.evaluate(generated_samples)

print(f"\n[Benchmark] Overall Similarity: {benchmark_results['overall_similarity']:.3f}")

print("\n[1] Distribution Similarity:")
dist_sim = benchmark_results['distribution_similarity']
print(f"  Score: {dist_sim['score']:.3f}")
if 'income' in dist_sim:
    income_dist = dist_sim['income']
    print(f"  Income JS-Divergence: {income_dist['js_divergence']:.4f}")
    print(f"  Income Similarity: {income_dist['similarity']:.3f}")
    print(f"    Generated: {income_dist['generated_dist']}")
    print(f"    Real:      {income_dist['real_dist']}")

print("\n[2] Statistical Similarity:")
stat_sim = benchmark_results['statistical_similarity']
print(f"  Score: {stat_sim['score']:.3f}")
if 'age' in stat_sim:
    age_stat = stat_sim['age']
    print(f"  Age Mean Similarity: {age_stat['mean_similarity']:.3f}")
    print(f"    Generated: {age_stat['generated']['mean']:.1f} ± {age_stat['generated']['std']:.1f}")
    print(f"    Real:      {age_stat['real']['mean']:.1f} ± {age_stat['real']['std']:.1f}")

print("\n[3] Logic Consistency:")
logic = benchmark_results['logic_consistency']
print(f"  Score: {logic['score']:.3f}")
print(f"  Consistent: {logic['consistent_count']}/{logic['total']}")

# Direct评估（Faithfulness + Diversity）
print("\n" + "=" * 80)
print("Direct Evaluation (Faithfulness + Diversity)")
print("=" * 80)

# 获取目标分布（用于Diversity评估）
target_dist = generator.scheduler.target_distribution

direct_results = generator.direct_evaluator.evaluate(generated_samples, target_dist)

print(f"\n[Direct] Overall Score: {direct_results['overall_score']:.3f}")

# Faithfulness（忠实度）
print(f"\n[1] Faithfulness (忠实度): {direct_results['faithfulness']['score']:.3f}")
constraint = direct_results['faithfulness']['constraint_check']
print(f"  Constraint Check:")
print(f"    - Valid: {constraint['valid_count']}/{constraint['total_count']}")
print(f"    - Score: {constraint['score']:.3f}")
if constraint.get('common_issues'):
    print(f"    - Common Issues: {constraint['common_issues']}")

# 如果有Benchmark评估
if 'benchmark_evaluation' in direct_results['faithfulness']:
    bench = direct_results['faithfulness']['benchmark_evaluation']
    print(f"  Benchmark (vs Real Data):")
    print(f"    - Overall Similarity: {bench['overall_similarity']:.3f}")

# Diversity（多样性）
print(f"\n[2] Diversity (多样性): {direct_results['diversity']['score']:.3f}")
diversity = direct_results['diversity']
print(f"  Vocabulary Diversity:")
print(f"    - Unique incomes: {diversity['unique_values']['income']}")
print(f"    - Unique educations: {diversity['unique_values']['education']}")
print(f"    - Unique occupations: {diversity['unique_values']['occupation']}")
print(f"  Sample Uniqueness: {diversity['uniqueness']:.3f}")
print(f"  Coverage: {diversity['coverage']:.3f}")

# 保存结果
df = pd.DataFrame(generated_samples)
df.to_csv("benchmark_test_output.csv", index=False)

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Generated Samples: {len(generated_samples)}")
print(f"Benchmark Similarity: {benchmark_results['overall_similarity']:.3f}")
print(f"Direct Quality Score: {direct_results['overall_score']:.3f}")
print(f"\nSaved to: benchmark_test_output.csv")

print("\n[Framework Complete]")
print("  ✓ Generation: Self-Instruct iterative generation")
print("  ✓ Curation: Filter + Auxiliary Model")
print("  ✓ Evaluation: Benchmark (vs real) + Direct")
