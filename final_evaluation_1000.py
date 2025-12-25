"""
最终评估：生成3000条样本并进行完整评估
使用判别器引导模式
"""

import sys
sys.path.append('generator_modules')

from generator_modules.main_generator import DataGenerator
import pandas as pd
import time
from datetime import datetime


def main():
    print("=" * 80)
    print("Adult Census 数据生成系统 - 最终评估")
    print("=" * 80)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标: 生成 3000 条样本")
    print(f"模式: 判别器引导 (主动学习)")
    print(f"评估: Benchmark + Direct (全维度)")
    
    start_time = time.time()
    
    # ========== 初始化生成器 ==========
    print("\n" + "=" * 80)
    print("Step 1: 初始化生成器")
    print("=" * 80)
    
    generator = DataGenerator(
        verbose=True,
        use_discriminator_guidance=True,
        discriminator_model_dir="adult_v2/trained_holistic_discriminator"
    )
    
    # ========== 加载真实数据 ==========
    print("\n" + "=" * 80)
    print("Step 2: 加载真实数据（全量，用于Benchmark评估）")
    print("=" * 80)
    
    generator.load_real_samples("archive/adult.csv", limit=None)
    
    # ========== 生成数据 ==========
    print("\n" + "=" * 80)
    print("Step 3: 生成 3000 条样本")
    print("=" * 80)
    print("\n提示: 生成过程可能需要较长时间（预计30-60分钟），请耐心等待...")
    print("进度提示: 每生成100条会显示一次进度\n")
    
    generation_start = time.time()
    
    # 分批生成，便于观察进度
    samples = []
    batch_size = 100
    total_target = 3000
    
    for i in range(0, total_target, batch_size):
        batch_num = i // batch_size + 1
        print(f"\n[Batch {batch_num}/{total_target//batch_size}] 生成 {batch_size} 条样本...")
        
        batch_samples = generator.generate_simple(n_samples=batch_size)
        samples.extend(batch_samples)
        
        print(f"  当前累计: {len(samples)}/{total_target} 条")
        
        if hasattr(generator, 'discriminator') and generator.discriminator and len(batch_samples) > 0:
            batch_scores = generator.discriminator.score_batch(batch_samples)
            print(f"  本批次平均真实度: {batch_scores.mean():.3f}")
    
    generation_time = time.time() - generation_start
    
    print(f"\n生成完成！")
    print(f"  实际生成: {len(samples)} 条样本")
    print(f"  耗时: {generation_time:.1f} 秒 ({generation_time/60:.1f} 分钟)")
    print(f"  速度: {len(samples)/generation_time:.2f} 样本/秒")
    
    if len(samples) == 0:
        print("\n⚠ 未生成任何样本，评估终止")
        return
    
    # ========== 整体判别器评分 ==========
    print("\n" + "=" * 80)
    print("Step 4: 整体判别器评估")
    print("=" * 80)
    
    if hasattr(generator, 'discriminator') and generator.discriminator:
        scores = generator.discriminator.score_batch(samples)
        
        print(f"\n真实度统计:")
        print(f"  样本数: {len(samples)}")
        print(f"  平均真实度: {scores.mean():.3f}")
        print(f"  真实度范围: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"  标准差: {scores.std():.3f}")
        
        print(f"\n质量分布:")
        print(f"  高质量 (>0.7):   {(scores > 0.7).sum()}/{len(scores)} ({(scores > 0.7).mean():.1%})")
        print(f"  中等质量 (0.5-0.7): {((scores >= 0.5) & (scores <= 0.7)).sum()}/{len(scores)} ({((scores >= 0.5) & (scores <= 0.7)).mean():.1%})")
        print(f"  低质量 (<0.5):   {(scores < 0.5).sum()}/{len(scores)} ({(scores < 0.5).mean():.1%})")
        
        # 调度器质量报告
        if hasattr(generator.scheduler, 'get_quality_report'):
            print(f"\n调度器质量分析:")
            report = generator.scheduler.get_quality_report()
            
            if 'category_quality' in report:
                print(f"\n  各类别平均真实度:")
                for dimension, scores_dict in report['category_quality'].items():
                    print(f"\n    {dimension}:")
                    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1])
                    for category, score in sorted_items:
                        quality_mark = "⭐" if score > 0.95 else "✓" if score > 0.90 else "⚠" if score > 0.85 else "✗"
                        print(f"      {quality_mark} {category}: {score:.3f}")
    
    # ========== Benchmark评估 ==========
    print("\n" + "=" * 80)
    print("Step 5: Benchmark评估 (vs 全量真实数据)")
    print("=" * 80)
    
    if hasattr(generator, 'benchmark_evaluator') and generator.benchmark_evaluator:
        benchmark_result = generator.benchmark_evaluator.evaluate(samples)
        
        print(f"\n综合相似度: {benchmark_result.get('overall_similarity', 0):.3f}")
        
        # 分布相似度
        dist_sim = benchmark_result.get('distribution_similarity', {})
        print(f"\n[1] 分布相似度 (权重40%): {dist_sim.get('score', 0):.3f}")
        for field in ['income', 'education', 'occupation', 'marital.status', 'sex']:
            if field in dist_sim:
                sim = dist_sim[field].get('similarity', 0)
                print(f"    {field}: {sim:.3f}")
        
        # 统计相似度
        stat_sim = benchmark_result.get('statistical_similarity', {})
        print(f"\n[2] 统计相似度 (权重30%): {stat_sim.get('score', 0):.3f}")
        for field in ['age', 'hours.per.week', 'education.num']:
            if field in stat_sim:
                mean_sim = stat_sim[field].get('mean_similarity', 0)
                std_sim = stat_sim[field].get('std_similarity', 0)
                print(f"    {field}: mean={mean_sim:.3f}, std={std_sim:.3f}")
        
        # 逻辑一致性
        logic_check = benchmark_result.get('logic_consistency', {})
        print(f"\n[3] 逻辑一致性 (权重30%): {logic_check.get('score', 0):.3f}")
        print(f"    一致样本: {logic_check.get('consistent_count', 0)}/{logic_check.get('total', 0)}")
    
    # ========== Direct评估 ==========
    print("\n" + "=" * 80)
    print("Step 6: Direct评估 (Faithfulness + Diversity)")
    print("=" * 80)
    
    if hasattr(generator, 'direct_evaluator') and generator.direct_evaluator:
        direct_result = generator.direct_evaluator.evaluate(samples)
        
        print(f"\n综合评分: {direct_result.get('overall_score', 0):.3f}")
        
        # Faithfulness
        faith = direct_result.get('faithfulness', {})
        print(f"\n[1] Faithfulness (权重60%): {faith.get('score', 0):.3f}")
        
        constraint = faith.get('constraint_check', {})
        print(f"    Constraint Check: {constraint.get('score', 0):.3f}")
        print(f"      有效样本: {constraint.get('valid_count', 0)}/{len(samples)}")
        
        if faith.get('benchmark_score') is not None:
            print(f"    Benchmark: {faith.get('benchmark_score', 0):.3f}")
        
        # Diversity
        div = direct_result.get('diversity', {})
        print(f"\n[2] Diversity (权重40%): {div.get('score', 0):.3f}")
        
        vocab = div.get('vocabulary_diversity', {})
        if vocab:
            print(f"    词汇多样性:")
            print(f"      unique incomes: {vocab.get('income', 0)}")
            print(f"      unique educations: {vocab.get('education', 0)}")
            print(f"      unique occupations: {vocab.get('occupation', 0)}")
        
        print(f"    样本独特性: {div.get('sample_uniqueness', 0):.3f}")
        print(f"    分布覆盖度: {div.get('coverage', 0):.3f}")
    
    # ========== 保存结果 ==========
    print("\n" + "=" * 80)
    print("Step 7: 保存结果")
    print("=" * 80)
    
    output_file = "final_evaluation_3000_samples.csv"
    df = pd.DataFrame(samples)
    df.to_csv(output_file, index=False)
    
    print(f"\n已保存: {output_file}")
    print(f"  样本数: {len(samples)}")
    print(f"  字段数: {len(df.columns)}")
    
    # ========== 总结 ==========
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("最终评估完成")
    print("=" * 80)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    
    print(f"\n核心指标总结:")
    if hasattr(generator, 'discriminator') and generator.discriminator:
        print(f"  平均真实度: {scores.mean():.3f}")
        print(f"  高质量比例: {(scores > 0.7).mean():.1%}")
    
    if hasattr(generator, 'benchmark_evaluator') and generator.benchmark_evaluator:
        print(f"  Benchmark相似度: {benchmark_result.get('overall_similarity', 0):.3f}")
    
    if hasattr(generator, 'direct_evaluator') and generator.direct_evaluator:
        print(f"  Direct综合评分: {direct_result.get('overall_score', 0):.3f}")
        print(f"  Faithfulness: {faith.get('score', 0):.3f}")
        print(f"  Diversity: {div.get('score', 0):.3f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
