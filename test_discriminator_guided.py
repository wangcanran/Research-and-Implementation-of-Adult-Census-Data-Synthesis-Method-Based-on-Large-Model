"""
测试判别器引导的Scheduler
对比：普通调度器 vs 判别器引导调度器
"""

import sys
sys.path.append('generator_modules')

from generator_modules.main_generator import DataGenerator
import pandas as pd


def test_discriminator_guided_generation():
    """测试判别器引导的生成"""
    
    print("=" * 80)
    print("判别器引导的数据生成测试")
    print("=" * 80)
    
    # ========== 模式1: 普通调度器 ==========
    print("\n【模式1】普通调度器（基于目标分布）")
    print("-" * 80)
    
    generator_normal = DataGenerator(
        verbose=True,
        use_discriminator_guidance=False  # 不使用判别器
    )
    
    generator_normal.load_real_samples("archive/adult.csv", limit=None)
    
    normal_samples = generator_normal.generate_simple(n_samples=50)
    
    print(f"\n生成结果: {len(normal_samples)} 个样本")
    
    # ========== 模式2: 判别器引导调度器 ==========
    print("\n" + "=" * 80)
    print("【模式2】判别器引导调度器（主动学习）")
    print("-" * 80)
    
    generator_guided = DataGenerator(
        verbose=True,
        use_discriminator_guidance=True,  # 使用判别器引导
        discriminator_model_dir="adult_v2/trained_holistic_discriminator"
    )
    
    generator_guided.load_real_samples("archive/adult.csv", limit=None)
    
    guided_samples = generator_guided.generate_simple(n_samples=50)
    
    print(f"\n生成结果: {len(guided_samples)} 个样本")
    
    # ========== 对比质量 ==========
    print("\n" + "=" * 80)
    print("质量对比")
    print("=" * 80)
    
    # 如果有判别器，对比两组样本的质量
    if hasattr(generator_guided, 'discriminator') and generator_guided.discriminator:
        print("\n使用整体判别器评估两组样本的真实度：")
        
        # 评估普通调度器生成的样本
        normal_scores = generator_guided.discriminator.score_batch(normal_samples)
        print(f"\n【普通调度器】")
        print(f"  平均真实度: {normal_scores.mean():.3f}")
        print(f"  真实度范围: [{normal_scores.min():.3f}, {normal_scores.max():.3f}]")
        print(f"  高质量比例 (>0.7): {(normal_scores > 0.7).mean():.1%}")
        
        # 评估判别器引导生成的样本
        guided_scores = generator_guided.discriminator.score_batch(guided_samples)
        print(f"\n【判别器引导】")
        print(f"  平均真实度: {guided_scores.mean():.3f}")
        print(f"  真实度范围: [{guided_scores.min():.3f}, {guided_scores.max():.3f}]")
        print(f"  高质量比例 (>0.7): {(guided_scores > 0.7).mean():.1%}")
        
        # 对比
        improvement = (guided_scores.mean() - normal_scores.mean()) / normal_scores.mean()
        print(f"\n【对比】")
        print(f"  真实度提升: {improvement:+.1%}")
        
        # 获取调度器的质量报告（如果支持）
        if hasattr(generator_guided.scheduler, 'get_quality_report'):
            print("\n【判别器引导调度器 - 质量报告】")
            report = generator_guided.scheduler.get_quality_report()
            print(f"  总样本数: {report.get('total_samples', 0)}")
            print(f"  平均真实度: {report.get('mean_score', 0):.3f}")
            
            if 'category_quality' in report:
                print(f"\n  各类别真实度:")
                for dimension, scores in report['category_quality'].items():
                    print(f"    {dimension}:")
                    for category, score in scores.items():
                        print(f"      {category}: {score}")
    else:
        print("\n⚠ 判别器不可用，跳过质量对比")
        print("  请先训练判别器:")
        print("    cd adult_v2")
        print("    python -c \"from adult_holistic_discriminator import train_holistic_discriminator; train_holistic_discriminator()\"")
    
    # ========== 保存结果 ==========
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    
    df_normal = pd.DataFrame(normal_samples)
    df_normal.to_csv("output_normal_scheduler.csv", index=False)
    print(f"  普通调度器: output_normal_scheduler.csv ({len(normal_samples)} 样本)")
    
    df_guided = pd.DataFrame(guided_samples)
    df_guided.to_csv("output_discriminator_guided.csv", index=False)
    print(f"  判别器引导: output_discriminator_guided.csv ({len(guided_samples)} 样本)")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_discriminator_guided_generation()
