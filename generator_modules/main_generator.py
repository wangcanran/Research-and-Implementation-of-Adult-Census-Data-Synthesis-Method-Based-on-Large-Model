"""
Main Generator Module
从 data_generator.py 提取，修改为 Adult Census
包含完整的 Self-Instruct 迭代生成策略
"""

import pandas as pd
from typing import List, Dict, Optional

from .task_spec import GenerationCondition
from .demonstration_manager import DemonstrationManager
from .decomposer import SampleWiseDecomposer
from .scheduler import DatasetWiseScheduler
from .curation import CurationPipeline, SampleFilter
from .evaluation import BenchmarkEvaluator, DirectEvaluator
from .discriminator_guided_scheduler import DiscriminatorGuidedScheduler, HybridScheduler
from .holistic_discriminator import load_discriminator, HAS_DISCRIMINATOR


class DataGenerator:
    """
    数据生成器（照搬data_generator.py的完整架构）
    
    核心功能：
    1. Self-Instruct 迭代生成（照搬）
    2. 三阶段框架：Generation → Curation → Evaluation
    3. 动态示例池（Self-Training）
    """
    
    def __init__(self, target_distribution: Optional[Dict] = None, verbose: bool = True,
                 use_discriminator_guidance: bool = False,
                 discriminator_model_dir: str = "adult_v2/trained_holistic_discriminator"):
        """
        初始化
        
        Args:
            target_distribution: 目标分布
            verbose: 是否输出详细信息
            use_discriminator_guidance: 是否使用判别器引导（需要已训练的判别器）
            discriminator_model_dir: 判别器模型目录
        """
        self.verbose = verbose
        self.use_discriminator_guidance = use_discriminator_guidance
        
        # 组件初始化（照搬data_generator.py）
        self.demo_manager = DemonstrationManager(use_heuristic=True)
        self.decomposer = SampleWiseDecomposer(demo_manager=self.demo_manager)
        
        # 调度器：判别器引导 vs 普通调度
        self.discriminator = None
        if use_discriminator_guidance and HAS_DISCRIMINATOR:
            self.discriminator = load_discriminator(discriminator_model_dir)
            if self.discriminator:
                self.scheduler = HybridScheduler(
                    discriminator=self.discriminator,
                    target_distribution=target_distribution
                )
                if self.verbose:
                    print("[Scheduler] 使用判别器引导调度器（主动学习）")
            else:
                self.scheduler = DatasetWiseScheduler(target_distribution)
                if self.verbose:
                    print("[Scheduler] 判别器加载失败，回退到普通调度器")
        else:
            self.scheduler = DatasetWiseScheduler(target_distribution)
            if self.verbose and use_discriminator_guidance:
                print("[Scheduler] 判别器不可用，使用普通调度器")
        
        self.curation_pipeline = CurationPipeline(use_filter=True, use_auxiliary=True)
        self.sample_filter = SampleFilter(strict=True)
        
        # 评估器（照搬data_generator.py）
        self.benchmark_evaluator = BenchmarkEvaluator()
        self.direct_evaluator = DirectEvaluator()
        
        # 迭代历史（照搬）
        self.iteration_history = []
    
    def load_real_samples(self, file_path: str, limit: int = None):
        """加载真实样本"""
        if self.verbose:
            print(f"\n[Loading] Real samples from {file_path}")
        
        df = pd.read_csv(file_path)
        if limit:
            df = df.sample(n=min(limit, len(df)), random_state=42)
        
        samples = df.to_dict('records')
        self.demo_manager.load_samples(samples)
        
        # 加载到Benchmark评估器（照搬data_generator.py）
        self.benchmark_evaluator.load_real_samples(samples)
        
        # 连接Benchmark到Direct Evaluator（用于Faithfulness评估，照搬data_generator.py）
        self.direct_evaluator.set_benchmark_evaluator(self.benchmark_evaluator)
        
        # 训练辅助模型（判别器机制，照搬data_generator.py）
        if self.curation_pipeline.auxiliary_model:
            if self.verbose:
                print(f"  Training auxiliary model...")
            self.curation_pipeline.train_auxiliary(samples)
        
        if self.verbose:
            print(f"  Loaded {len(samples)} samples")
            print(f"  Benchmark evaluator ready")
            print(f"  Direct evaluator configured (Faithfulness + Diversity)")
    
    def iterative_generate(self, seed_samples: List[Dict],
                          target_count: int,
                          n_iterations: int = 3,
                          quality_threshold: float = 0.6,
                          verbose: bool = True) -> List[Dict]:
        """
        Self-Instruct 迭代生成（完全照搬data_generator.py）
        
        Args:
            seed_samples: 种子样本
            target_count: 目标生成数量
            n_iterations: 迭代次数
            quality_threshold: 质量阈值
            verbose: 是否输出
        
        Returns:
            高质量生成样本列表
        """
        # 初始化：将种子样本加入示例池（照搬）
        self.demo_manager.real_samples = seed_samples.copy()
        all_generated = seed_samples.copy()
        
        if verbose:
            print(f"\n[Self-Instruct] Iterative Generation Started")
            print(f"  Seed samples: {len(seed_samples)}")
            print(f"  Target count: {target_count}")
            print(f"  Iterations: {n_iterations}")
        
        for iteration in range(n_iterations):
            if len(all_generated) >= target_count:
                break
            
            if verbose:
                print(f"\n  --- Iteration {iteration + 1}/{n_iterations} ---")
            
            # Step 1: 生成新样本（照搬）
            batch_size = min(target_count - len(all_generated),
                           max(50, target_count // n_iterations))
            
            new_samples = self._generate_batch(batch_size)
            
            if verbose:
                print(f"  Generated: {len(new_samples)} samples")
            
            # Step 2: 启发式过滤（照搬）
            passed, failed = self.sample_filter.filter_samples(new_samples)
            
            if verbose:
                print(f"  After filter: {len(passed)} samples (filtered {len(failed)})")
            
            # Step 3: 质量检查 - 只保留高质量样本（照搬）
            high_quality = []
            for sample in passed:
                score, _ = self.sample_filter.evaluate_sample(sample)
                if score >= quality_threshold:
                    high_quality.append(sample)
            
            if verbose:
                print(f"  High quality: {len(high_quality)} samples (>= {quality_threshold})")
            
            # Step 4: 更新示例池（Self-Training）（照搬）
            if high_quality:
                # 选择最高质量的样本加入示例池
                quality_scores = [(s, self.sample_filter.evaluate_sample(s)[0])
                                 for s in high_quality]
                quality_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 只加入top 20%或最多50个（照搬）
                n_to_add = min(len(high_quality) // 5, 50)
                new_demos = [s for s, _ in quality_scores[:n_to_add]]
                
                self.demo_manager.real_samples.extend(new_demos)
                all_generated.extend(high_quality)
                
                if verbose:
                    print(f"  Added to demo pool: {len(new_demos)} samples")
                    print(f"  Total accumulated: {len(all_generated)} samples")
            
            # 记录迭代结果（照搬）
            self.iteration_history.append({
                "iteration": iteration + 1,
                "generated": len(new_samples),
                "passed_filter": len(passed),
                "high_quality": len(high_quality),
                "total_accumulated": len(all_generated)
            })
        
        if verbose:
            print(f"\n[Self-Instruct] Complete, total: {len(all_generated)} high-quality samples")
        
        return all_generated[:target_count]
    
    def _generate_batch(self, batch_size: int) -> List[Dict]:
        """生成一批样本（照搬data_generator.py）"""
        samples = []
        batch_samples = []
        
        for _ in range(batch_size):
            try:
                # 使用调度器获取条件
                condition = self.scheduler.get_next_condition()
                
                sample = self.decomposer.decompose_and_generate(condition)
                samples.append(sample)
                batch_samples.append(sample)
                
                # 批量更新缓存和调度器
                if len(batch_samples) >= 10:
                    all_samples = samples.copy()
                    all_samples.extend(batch_samples)
                    
                    # 批量更新缓存
                    for sample in batch_samples:
                        self.decomposer.update_cache(sample)
                    
                    # 批量更新调度器（判别器引导模式需要评分）
                    if self.use_discriminator_guidance and self.discriminator:
                        # 判别器引导模式：批量评分
                        scores = self.discriminator.score_batch(batch_samples)
                        for sample, score in zip(batch_samples, scores):
                            self.scheduler.update(sample, score)
                    else:
                        # 普通模式
                        for sample in batch_samples:
                            self.scheduler.update(sample)
                    
                    batch_samples = []
            except Exception as e:
                if self.verbose:
                    print(f"    [Warning] Sample generation failed: {e}")
                continue
        
        # 处理剩余的样本
        if batch_samples:
            all_samples = samples.copy()
            all_samples.extend(batch_samples)
            
            # 批量更新缓存
            for sample in batch_samples:
                self.decomposer.update_cache(sample)
            
            # 批量更新调度器（判别器引导模式需要评分）
            if self.use_discriminator_guidance and self.discriminator:
                # 判别器引导模式：批量评分
                scores = self.discriminator.score_batch(batch_samples)
                for sample, score in zip(batch_samples, scores):
                    self.scheduler.update(sample, score)
            else:
                # 普通模式
                for sample in batch_samples:
                    self.scheduler.update(sample)
        
        return samples
    
    def get_iteration_summary(self) -> Dict:
        """获取迭代过程统计（照搬data_generator.py）"""
        if not self.iteration_history:
            return {}
        
        return {
            "total_iterations": len(self.iteration_history),
            "total_generated": sum(h["generated"] for h in self.iteration_history),
            "total_high_quality": sum(h["high_quality"] for h in self.iteration_history),
            "avg_pass_rate": sum(h["passed_filter"] / h["generated"]
                                 for h in self.iteration_history if h["generated"] > 0) / len(self.iteration_history),
            "iterations": self.iteration_history
        }
    
    def generate_simple(self, n_samples: int) -> List[Dict]:
        """简单生成（不使用Self-Instruct）"""
        if self.verbose:
            print(f"\n[Simple Generation] Generating {n_samples} samples...")
        
        samples = self._generate_batch(n_samples)
        
        # 策展
        result = self.curation_pipeline.curate(samples)
        
        if self.verbose:
            print(f"  Generated: {len(samples)}")
            print(f"  After curation: {len(result['curated_samples'])}")
        
        return result['curated_samples']
    
    def save_to_csv(self, samples: List[Dict], file_path: str):
        """保存到CSV"""
        df = pd.DataFrame(samples)
        df.to_csv(file_path, index=False)
        if self.verbose:
            print(f"\n[Saved] {len(samples)} samples to {file_path}")


# ============================================================================
#                      快速生成函数（照搬data_generator.py）
# ============================================================================

def quick_generate(n_samples: int = 100,
                  data_file: str = "archive/adult.csv",
                  use_self_instruct: bool = True,
                  n_iterations: int = 3) -> List[Dict]:
    """
    快速生成（照搬data_generator.py）
    
    Args:
        n_samples: 生成数量
        data_file: 真实数据文件
        use_self_instruct: 是否使用Self-Instruct
        n_iterations: 迭代次数
    
    Returns:
        生成样本列表
    """
    generator = DataGenerator(verbose=True)
    
    # 加载真实数据
    generator.load_real_samples(data_file, limit=1000)
    
    if use_self_instruct:
        # 使用种子样本（前10个真实样本）
        seed_samples = generator.demo_manager.real_samples[:10]
        
        # Self-Instruct 迭代生成
        samples = generator.iterative_generate(
            seed_samples=seed_samples,
            target_count=n_samples,
            n_iterations=n_iterations,
            quality_threshold=0.6
        )
    else:
        # 简单生成
        samples = generator.generate_simple(n_samples)
    
    return samples
