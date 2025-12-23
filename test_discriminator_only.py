"""
仅测试判别器引导的生成
"""

import sys
sys.path.append('generator_modules')

from generator_modules.main_generator import DataGenerator
import pandas as pd


def test_discriminator_only():
    """只测试判别器引导模式"""
    
    print("=" * 80)
    print("判别器引导生成测试")
    print("=" * 80)
    
    generator = DataGenerator(
        verbose=True,
        use_discriminator_guidance=True,
        discriminator_model_dir="adult_v2/trained_holistic_discriminator"
    )
    
    print("\n[Loading] 加载真实样本（全量数据用于Benchmark评估）...")
    generator.load_real_samples("archive/adult.csv", limit=None)
    
    print("\n[Generation] 生成样本...")
    samples = generator.generate_simple(n_samples=50)
    
    print(f"\n生成结果: {len(samples)} 个样本")
    
    if len(samples) == 0:
        print("\n⚠ 未生成任何样本")
        return
    
    # 评估质量
    if hasattr(generator, 'discriminator') and generator.discriminator:
        print("\n" + "=" * 80)
        print("质量评估")
        print("=" * 80)
        
        scores = generator.discriminator.score_batch(samples)
        
        print(f"\n真实度统计:")
        print(f"  平均真实度: {scores.mean():.3f}")
        print(f"  真实度范围: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"  高质量样本 (>0.7): {(scores > 0.7).sum()}/{len(scores)} ({(scores > 0.7).mean():.1%})")
        print(f"  中等质量样本 (0.5-0.7): {((scores >= 0.5) & (scores <= 0.7)).sum()}/{len(scores)}")
        print(f"  低质量样本 (<0.5): {(scores < 0.5).sum()}/{len(scores)}")
        
        # 获取调度器质量报告
        if hasattr(generator.scheduler, 'get_quality_report'):
            print("\n" + "=" * 80)
            print("调度器质量报告")
            print("=" * 80)
            
            report = generator.scheduler.get_quality_report()
            print(f"\n总样本数: {report.get('total_samples', 0)}")
            print(f"平均真实度: {report.get('mean_score', 0):.3f}")
            
            if 'category_quality' in report:
                print(f"\n各类别真实度:")
                for dimension, scores_dict in report['category_quality'].items():
                    print(f"\n  {dimension}:")
                    for category, score in scores_dict.items():
                        print(f"    {category}: {score:.3f}")
    
    # 保存结果
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    
    df = pd.DataFrame(samples)
    df.to_csv("output_discriminator_only.csv", index=False)
    print(f"已保存: output_discriminator_only.csv ({len(samples)} 样本)")
    
    # 显示部分样本
    print("\n示例样本 (前5个):")
    print(df.head().to_string(index=False))
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_discriminator_only()
