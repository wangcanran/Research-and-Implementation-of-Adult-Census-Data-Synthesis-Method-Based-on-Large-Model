#!/usr/bin/env python3
"""测试改进版调度器：对比质量与多样性"""

import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from generator_modules.diversity_enhanced_scheduler import DiversityEnhancedScheduler
from generator_modules.decomposer import SampleWiseDecomposer
from generator_modules.demonstration_manager import DemonstrationManager
from generator_modules.evaluation import BenchmarkEvaluator, DirectEvaluator
from generator_modules.holistic_discriminator import load_discriminator
import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_real_data():
    """加载真实数据"""
    df = pd.read_csv("archive/adult.csv")
    samples = df.to_dict('records')
    logger.info(f"Loaded {len(samples)} real samples")
    return samples


def create_target_distribution():
    """创建目标分布"""
    return {
        "income": {
            "<=50K": 0.76,
            ">50K": 0.24
        },
        "age_range": {
            "young": 0.25,
            "middle": 0.55,
            "senior": 0.20
        },
        "education_level": {
            "low": 0.30,
            "medium": 0.50,
            "high": 0.20
        }
    }


def generate_samples(num_samples: int = 500):
    """
    使用改进版调度器生成样本
    
    参数:
        num_samples: 生成样本数（默认500，用于快速测试）
    """
    logger.info("="*80)
    logger.info("IMPROVED SCHEDULER TEST")
    logger.info("="*80)
    
    # 1. 加载真实数据
    logger.info("\nStep 1: Loading real data...")
    real_samples = load_real_data()
    
    # 2. 初始化组件
    logger.info("\nStep 2: Initializing components...")
    
    # 改进版调度器
    target_dist = create_target_distribution()
    scheduler = DiversityEnhancedScheduler(
        target_distribution=target_dist,
        discriminator_ratio=0.05,    # 5% 判别器引导
        rare_sampling_ratio=0.05     # 5% 稀有值采样
    )
    
    # 判别器（加载已训练好的模型）
    logger.info("Loading pre-trained discriminator...")
    discriminator = load_discriminator("adult_v2/trained_holistic_discriminator")
    
    if discriminator is None:
        logger.error("Failed to load discriminator. Please ensure it's trained.")
        return []
    
    logger.info("Discriminator loaded successfully")
    
    # 示例管理器
    demo_manager = DemonstrationManager(use_heuristic=True)
    demo_manager.load_samples(real_samples)
    
    # 分解器（传入目标分布，启用自适应校正）
    decomposer = SampleWiseDecomposer(demo_manager, target_distribution=target_dist)
    
    # 3. 生成样本
    logger.info(f"\nStep 3: Generating {num_samples} samples...")
    logger.info("Strategy: 90% Distribution + 5% Discriminator + 5% Rare Sampling")
    
    generated_samples = []
    high_quality_count = 0
    
    batch_size = 10
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        # 选择生成条件
        condition = scheduler.select_next_condition()
        
        # 选择示例
        demonstrations = demo_manager.select_demonstrations(k=3, condition=condition)
        
        # 生成样本
        try:
            # decomposer只能一次生成一个样本
            batch_samples = []
            for _ in range(min(batch_size, num_samples - len(generated_samples))):
                sample = decomposer.decompose_and_generate(condition)
                if sample:
                    batch_samples.append(sample)
            
            # 判别器评分
            for sample in batch_samples:
                score = discriminator.score(sample)
                
                if score > 0.7:  # 高质量阈值
                    generated_samples.append(sample)
                    high_quality_count += 1
                    scheduler.add_sample(sample, score)
                    
                    # 【关键】更新生成缓存
                    decomposer.update_cache(sample)
                    
                    # 【关键】每50个样本重置缓存，防止累积偏差
                    if len(generated_samples) % 50 == 0:
                        decomposer.reset_cache()
                        logger.info(f"  [Cache Reset] Reset at {len(generated_samples)} samples to prevent distribution drift")
                else:
                    logger.warning(f"Low quality sample rejected (score={score:.3f})")
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Progress: {len(generated_samples)}/{num_samples} samples "
                          f"(quality: {high_quality_count/len(generated_samples)*100:.1f}%)")
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
        
        if len(generated_samples) >= num_samples:
            break
    
    logger.info(f"\nGeneration complete: {len(generated_samples)} samples")
    if len(generated_samples) > 0:
        logger.info(f"High quality rate: {high_quality_count/len(generated_samples)*100:.1f}%")
    else:
        logger.warning("No samples generated!")
        return []
    
    # 4. 保存结果
    output_file = f"improved_scheduler_{num_samples}_samples.csv"
    df = pd.DataFrame(generated_samples)
    df.to_csv(output_file, index=False)
    logger.info(f"\nSaved to: {output_file}")
    
    # 5. 评估
    logger.info("\n" + "="*80)
    logger.info("EVALUATION")
    logger.info("="*80)
    
    # 多样性评估
    evaluate_diversity(generated_samples)
    
    # Benchmark评估
    logger.info("\nBenchmark Evaluation:")
    benchmark_eval = BenchmarkEvaluator(real_samples)
    benchmark_results = benchmark_eval.evaluate(generated_samples)
    
    logger.info(f"  Overall similarity: {benchmark_results['overall_similarity']*100:.1f}%")
    logger.info(f"  Distribution similarity: {benchmark_results['distribution_similarity']['score']*100:.1f}%")
    logger.info(f"  Logic consistency: {benchmark_results['logic_consistency']['score']*100:.1f}%")
    
    # Direct评估
    logger.info("\nDirect Evaluation:")
    direct_eval = DirectEvaluator()
    direct_results = direct_eval.evaluate(generated_samples, target_dist)
    
    logger.info(f"  Faithfulness: {direct_results['faithfulness']['score']*100:.1f}%")
    logger.info(f"  Diversity: {direct_results['diversity']['score']*100:.1f}%")
    logger.info(f"  Uniqueness: {direct_results['diversity']['uniqueness']*100:.1f}%")
    
    return generated_samples


def evaluate_diversity(samples):
    """评估多样性指标"""
    logger.info("\nDiversity Analysis:")
    
    # 1. 组合独特性
    signatures = []
    for s in samples:
        sig = f"{s.get('age')}_{s.get('education')}_{s.get('occupation')}_{s.get('income')}"
        signatures.append(sig)
    
    unique_combos = len(set(signatures))
    uniqueness = unique_combos / len(signatures) if signatures else 0
    
    logger.info(f"  Total samples: {len(samples)}")
    logger.info(f"  Unique combinations: {unique_combos}")
    logger.info(f"  Uniqueness rate: {uniqueness*100:.1f}%")
    logger.info(f"  Average copies per combo: {len(signatures)/unique_combos:.2f}")
    
    # 2. 字段多样性
    unique_values = {
        "income": len(set(s.get("income") for s in samples if s.get("income"))),
        "education": len(set(s.get("education") for s in samples if s.get("education"))),
        "occupation": len(set(s.get("occupation") for s in samples if s.get("occupation"))),
        "age": len(set(s.get("age") for s in samples if s.get("age")))
    }
    
    logger.info(f"\n  Unique values per field:")
    for field, count in unique_values.items():
        logger.info(f"    {field}: {count}")


def compare_with_baseline():
    """对比改进前后的结果"""
    logger.info("\n" + "="*80)
    logger.info("COMPARISON WITH BASELINE")
    logger.info("="*80)
    
    baseline_file = "final_evaluation_3000_samples.csv"
    improved_file = "improved_scheduler_500_samples.csv"
    
    try:
        baseline_df = pd.read_csv(baseline_file)
        improved_df = pd.read_csv(improved_file)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return
    
    logger.info(f"\nBaseline (Original Scheduler):")
    logger.info(f"  Samples: {len(baseline_df)}")
    
    # Baseline diversity
    baseline_sigs = []
    for _, row in baseline_df.iterrows():
        sig = f"{row.get('age')}_{row.get('education')}_{row.get('occupation')}_{row.get('income')}"
        baseline_sigs.append(sig)
    baseline_unique = len(set(baseline_sigs))
    baseline_uniqueness = baseline_unique / len(baseline_sigs)
    
    logger.info(f"  Unique combinations: {baseline_unique}")
    logger.info(f"  Uniqueness rate: {baseline_uniqueness*100:.1f}%")
    
    logger.info(f"\nImproved (Diversity Enhanced):")
    logger.info(f"  Samples: {len(improved_df)}")
    
    # Improved diversity
    improved_sigs = []
    for _, row in improved_df.iterrows():
        sig = f"{row.get('age')}_{row.get('education')}_{row.get('occupation')}_{row.get('income')}"
        improved_sigs.append(sig)
    improved_unique = len(set(improved_sigs))
    improved_uniqueness = improved_unique / len(improved_sigs)
    
    logger.info(f"  Unique combinations: {improved_unique}")
    logger.info(f"  Uniqueness rate: {improved_uniqueness*100:.1f}%")
    
    logger.info(f"\nImprovement:")
    logger.info(f"  Uniqueness: {baseline_uniqueness*100:.1f}% → {improved_uniqueness*100:.1f}% "
               f"({(improved_uniqueness-baseline_uniqueness)*100:+.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test improved scheduler")
    parser.add_argument("--samples", type=int, default=500, 
                       help="Number of samples to generate (default: 500)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with baseline after generation")
    
    args = parser.parse_args()
    
    # 生成样本
    generate_samples(num_samples=args.samples)
    
    # 对比
    if args.compare:
        compare_with_baseline()
    
    logger.info("\nTest complete!")
