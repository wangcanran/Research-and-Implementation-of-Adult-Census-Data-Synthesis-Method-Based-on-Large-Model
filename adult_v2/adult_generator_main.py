"""
Adult Census Data Generator - Main Generator
主生成器 - 整合三阶段框架
"""

from typing import List, Dict, Optional
import pandas as pd

from .adult_task_spec import GenerationCondition
from .adult_demonstration_manager import AdultDemonstrationManager
from .adult_decomposer import AdultSampleWiseDecomposer
from .adult_scheduler import AdultDatasetWiseScheduler
from .adult_curation import AdultCurationPipeline
from .adult_evaluation import (
    AdultDirectEvaluator, AdultBenchmarkEvaluator, AdultIndirectEvaluator
)
from .adult_learner import AdultStatisticalLearner


class AdultDataGenerator:
    """
    Adult Census Data Generator (V2)
    
    完整的三阶段框架：
    I. Generation - 生成阶段
    II. Curation - 策展阶段
    III. Evaluation - 评估阶段
    """
    
    def __init__(self, target_distribution: Optional[Dict] = None, 
                 use_advanced_features: bool = True):
        """
        Args:
            target_distribution: 目标分布
            use_advanced_features: 是否使用高级功能（启发式选择、Curation、Evaluation）
        """
        # 统计学习器
        self.learner = AdultStatisticalLearner()
        self.learned_stats = {}
        
        # ========== I. Generation 组件 ==========
        self.demo_manager = AdultDemonstrationManager(use_heuristic=use_advanced_features)
        self.decomposer = None  # 会在学习统计后初始化
        self.scheduler = AdultDatasetWiseScheduler(target_distribution)
        
        # ========== II. Curation 组件 ==========
        self.curation_pipeline = AdultCurationPipeline(
            use_reweighting=use_advanced_features,
            use_enhancement=use_advanced_features
        )
        
        # ========== III. Evaluation 组件 ==========
        self.direct_evaluator = AdultDirectEvaluator()
        self.benchmark_evaluator = AdultBenchmarkEvaluator()
        self.indirect_evaluator = AdultIndirectEvaluator()
        
        # 连接Benchmark Evaluator到Direct Evaluator
        self.direct_evaluator.set_benchmark_evaluator(self.benchmark_evaluator)
        
        self.use_advanced_features = use_advanced_features
        self.verbose = True
    
    def load_real_samples(self, file_path: str, limit: int = 1000,
                         evaluation_limit: int = None):
        """
        加载真实样本数据
        
        用途：
        1. 学习统计特征（条件分布）
        2. 作为Demonstration示例
        3. 作为Benchmark评估基准
        
        Args:
            file_path: CSV文件路径
            limit: 用于学习和示例的样本数
            evaluation_limit: 用于评估的样本数（如果为None，使用全部数据）
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("加载真实数据")
            print("=" * 80)
        
        # 1. 学习统计特征
        self.learned_stats = self.learner.learn_from_data(file_path)
        
        # 2. 初始化Decomposer（需要learned_stats）
        self.decomposer = AdultSampleWiseDecomposer(learned_stats=self.learned_stats)
        
        # 3. 加载样本作为Demonstration
        self.demo_manager.load_samples_from_file(file_path)
        
        # 4. 加载样本作为Benchmark
        df = pd.read_csv(file_path)
        if evaluation_limit:
            benchmark_samples = df.head(evaluation_limit).to_dict('records')
        else:
            benchmark_samples = df.to_dict('records')
        self.benchmark_evaluator.load_real_samples(benchmark_samples)
        
        if self.verbose:
            print(f"\n[完成] 真实数据加载完成")
            print("=" * 80)
    
    def generate_batch(self, n_samples: int,
                      condition: Optional[GenerationCondition] = None,
                      use_scheduler: bool = True,
                      max_retries: int = 3) -> List[Dict]:
        """
        批量生成样本（仅Generation阶段）
        
        Args:
            n_samples: 生成样本数
            condition: 生成条件（如果为None且use_scheduler=True，则使用调度器）
            use_scheduler: 是否使用调度器（基于目标分布）
            max_retries: 每个样本最大重试次数
        
        Returns:
            生成的样本列表
        """
        if self.verbose:
            print(f"\n[Generation] 生成 {n_samples} 个样本...")
        
        samples = []
        self.decomposer.reset_cache()
        
        for i in range(n_samples):
            if self.verbose and (i + 1) % 50 == 0:
                print(f"  进度: {i + 1}/{n_samples}")
            
            # 确定生成条件
            if use_scheduler and condition is None:
                gen_condition = self.scheduler.get_next_condition()
            else:
                gen_condition = condition if condition else GenerationCondition()
            
            # 生成样本
            sample = None
            for retry in range(max_retries):
                try:
                    sample = self.decomposer.decompose_and_generate(gen_condition)
                    
                    # 简单验证
                    from .adult_task_spec import validate_sample
                    is_valid, errors = validate_sample(sample)
                    
                    if is_valid:
                        break
                    elif retry == max_retries - 1 and self.verbose:
                        print(f"  [警告] 样本{i+1}验证失败: {errors[:2]}")
                except Exception as e:
                    if retry == max_retries - 1 and self.verbose:
                        print(f"  [错误] 样本{i+1}生成失败: {e}")
            
            if sample:
                samples.append(sample)
                self.decomposer.update_cache(sample)
                
                # 更新调度器统计
                if use_scheduler:
                    self.scheduler.update(sample)
        
        if self.verbose:
            print(f"  [完成] 成功生成 {len(samples)}/{n_samples} 个样本")
        
        return samples
    
    def generate_with_curation(self, n_samples: int,
                               condition: Optional[GenerationCondition] = None,
                               use_scheduler: bool = True) -> Dict:
        """
        生成并策展（Generation + Curation）
        
        Returns:
            {
                "generated_samples": List[Dict],
                "curated_samples": List[Dict],
                "curation_stats": Dict,
                "sample_weights": Dict
            }
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("生成 + 策展流程")
            print("=" * 80)
        
        # I. Generation
        generated_samples = self.generate_batch(
            n_samples=n_samples,
            condition=condition,
            use_scheduler=use_scheduler
        )
        
        # II. Curation
        if self.verbose:
            print(f"\n[Curation] 开始策展...")
        
        curation_result = self.curation_pipeline.curate(generated_samples)
        
        if self.verbose:
            print(f"  原始样本: {curation_result['original_count']}")
            print(f"  过滤后: {curation_result['filter_stats']['passed']}")
            print(f"  最终样本: {curation_result['final_count']}")
            print(f"  通过率: {curation_result['filter_stats']['pass_rate']:.1%}")
        
        return {
            "generated_samples": generated_samples,
            "curated_samples": curation_result["curated_samples"],
            "curation_stats": curation_result,
            "sample_weights": curation_result.get("sample_weights")
        }
    
    def generate_with_full_pipeline(self, n_samples: int,
                                    condition: Optional[GenerationCondition] = None,
                                    use_scheduler: bool = True) -> Dict:
        """
        完整三阶段流程（Generation + Curation + Evaluation）
        
        Returns:
            {
                "generated_samples": List[Dict],
                "curated_samples": List[Dict],
                "curation_stats": Dict,
                "evaluation_results": Dict
            }
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("完整三阶段流程")
            print("=" * 80)
        
        # I. Generation + II. Curation
        result = self.generate_with_curation(
            n_samples=n_samples,
            condition=condition,
            use_scheduler=use_scheduler
        )
        
        # III. Evaluation
        if self.verbose:
            print(f"\n[Evaluation] 开始评估...")
        
        evaluation_results = self.direct_evaluator.evaluate(result["curated_samples"])
        
        if self.verbose:
            print(f"  格式正确率: {evaluation_results['format_correctness']['validity_rate']:.1%}")
            print(f"  逻辑一致率: {evaluation_results['logical_consistency']['consistency_rate']:.1%}")
            if 'faithfulness' in evaluation_results:
                print(f"  分布相似度: {evaluation_results['faithfulness']['overall_similarity']:.1%}")
            print(f"  总体评分: {evaluation_results['overall_score']:.1%}")
        
        result["evaluation_results"] = evaluation_results
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("流程完成")
            print("=" * 80)
        
        return result
    
    def save_to_csv(self, samples: List[Dict], output_file: str):
        """保存样本到CSV文件"""
        df = pd.DataFrame(samples)
        
        # 确保列顺序与原始数据一致
        column_order = [
            'age', 'workclass', 'fnlwgt', 'education', 'education.num',
            'marital.status', 'occupation', 'relationship', 'race', 'sex',
            'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income'
        ]
        
        # 只保留存在的列
        existing_cols = [col for col in column_order if col in df.columns]
        df = df[existing_cols]
        
        df.to_csv(output_file, index=False)
        
        if self.verbose:
            print(f"\n[保存] 已保存到 {output_file}")
    
    def get_scheduler_stats(self) -> Dict:
        """获取调度器的分布统计"""
        return {
            "current_distribution": self.scheduler.get_current_stats(),
            "target_distribution": self.scheduler.target_distribution,
            "distribution_gap": self.scheduler.get_distribution_gap()
        }
    
    def save_learned_stats(self, file_path: str):
        """保存学习的统计信息"""
        self.learner.save_stats(file_path)
    
    def load_learned_stats(self, file_path: str):
        """加载学习的统计信息"""
        self.learner.load_stats(file_path)
        self.learned_stats = self.learner.get_stats()
        # 重新初始化decomposer
        self.decomposer = AdultSampleWiseDecomposer(learned_stats=self.learned_stats)


# ============================================================================
#                      便捷函数
# ============================================================================

def quick_generate(data_file: str, n_samples: int = 100, 
                  output_file: str = "adult_synthetic.csv",
                  use_full_pipeline: bool = True) -> Dict:
    """
    快速生成样本的便捷函数
    
    Args:
        data_file: 真实数据文件路径
        n_samples: 生成样本数
        output_file: 输出文件路径
        use_full_pipeline: 是否使用完整流程（包含Curation和Evaluation）
    
    Returns:
        生成结果
    """
    # 初始化生成器
    generator = AdultDataGenerator(use_advanced_features=True)
    
    # 加载真实数据
    generator.load_real_samples(data_file, limit=1000)
    
    # 生成
    if use_full_pipeline:
        result = generator.generate_with_full_pipeline(n_samples=n_samples)
        samples = result["curated_samples"]
    else:
        samples = generator.generate_batch(n_samples=n_samples)
        result = {"generated_samples": samples}
    
    # 保存
    generator.save_to_csv(samples, output_file)
    
    return result
