#!/usr/bin/env python3
"""
改进版调度器：平衡质量与多样性

核心改进：
1. 降低判别器引导比例（20% → 5%）
2. 增加去重机制（避免重复组合）
3. 强制稀有值采样（探索低频组合）
4. 动态调整策略（根据多样性指标）
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
import random
import logging

from .scheduler import GenerationCondition, DatasetWiseScheduler

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """多样性指标"""
    unique_combinations: int
    total_samples: int
    uniqueness_rate: float
    coverage_rate: float  # 相对于目标分布的覆盖率
    

class DiversityEnhancedScheduler(DatasetWiseScheduler):
    """
    多样性增强调度器
    
    策略权重调整：
    - 分布探索：90% (原80%)
    - 判别器引导：5% (原20%)  ← 关键改进
    - 稀有值强制采样：5% (新增)
    """
    
    def __init__(self, 
                 target_distribution: Dict,
                 discriminator_ratio: float = 0.05,  # 降低判别器权重
                 rare_sampling_ratio: float = 0.05):   # 稀有值采样
        super().__init__(target_distribution)
        
        self.discriminator_ratio = discriminator_ratio
        self.rare_sampling_ratio = rare_sampling_ratio
        
        # 记录已生成的样本和评分（用于判别器引导）
        self.generated_samples: List[Dict] = []
        self.sample_scores: List[float] = []
        
        # 字段值频率统计
        self.field_value_counts: Dict[str, Counter] = defaultdict(Counter)
        
        # 多样性指标
        self.diversity_metrics = DiversityMetrics(0, 0, 0.0, 0.0)
        
        logger.info(f"DiversityEnhancedScheduler initialized:")
        logger.info(f"  - Discriminator ratio: {discriminator_ratio*100:.1f}%")
        logger.info(f"  - Rare sampling ratio: {rare_sampling_ratio*100:.1f}%")
        logger.info(f"  - Adaptive cache enabled (reset every 50 samples)")
    
    def _create_combination_signature(self, sample: Dict) -> str:
        """
        创建样本签名（用于多样性统计）
        
        基于关键字段：age, education, occupation, income
        """
        parts = [
            str(sample.get('age', 'unknown')),
            str(sample.get('education', 'unknown')),
            str(sample.get('occupation', 'unknown')),
            str(sample.get('income', 'unknown'))
        ]
        return "_".join(parts)
    
    def _update_diversity_metrics(self):
        """更新多样性指标"""
        if not self.generated_samples:
            return
        
        # 计算样本级别的独特组合数（基于关键字段）
        signatures = set()
        for sample in self.generated_samples:
            sig = self._create_combination_signature(sample)
            signatures.add(sig)
        
        unique = len(signatures)
        total = len(self.generated_samples)
        
        self.diversity_metrics = DiversityMetrics(
            unique_combinations=unique,
            total_samples=total,
            uniqueness_rate=unique / total if total > 0 else 0,
            coverage_rate=self._calculate_coverage_rate()
        )
    
    def _calculate_coverage_rate(self) -> float:
        """计算目标分布覆盖率"""
        if not self.target_distribution or not self.generated_stats:
            return 0.0
        
        covered = 0
        total = 0
        
        for field, target_dist in self.target_distribution.items():
            if field not in self.generated_stats:
                continue
            
            target_values = set(target_dist.keys())
            actual_values = set(self.generated_stats[field].keys())
            
            covered += len(target_values & actual_values)
            total += len(target_values)
        
        return covered / total if total > 0 else 0.0
    
    def _select_rare_values(self) -> GenerationCondition:
        """
        稀有值强制采样策略
        
        选择当前生成频率最低的值组合，强制探索
        """
        # 找出每个字段的最低频率值
        rare_values = {}
        
        # 收入
        if "income" in self.target_distribution:
            income_counts = self.field_value_counts.get("income", Counter())
            if income_counts:
                rare_values["income"] = income_counts.most_common()[-1][0]
            else:
                rare_values["income"] = random.choice(list(self.target_distribution["income"].keys()))
        
        # 年龄段
        if "age_range" in self.target_distribution:
            age_counts = self.field_value_counts.get("age_range", Counter())
            if age_counts:
                rare_values["age_range"] = age_counts.most_common()[-1][0]
            else:
                rare_values["age_range"] = random.choice(list(self.target_distribution["age_range"].keys()))
        
        # 教育水平
        if "education_level" in self.target_distribution:
            edu_counts = self.field_value_counts.get("education_level", Counter())
            if edu_counts:
                rare_values["education_level"] = edu_counts.most_common()[-1][0]
            else:
                rare_values["education_level"] = random.choice(list(self.target_distribution["education_level"].keys()))
        
        condition = GenerationCondition(
            income=rare_values.get("income"),
            age_range=rare_values.get("age_range"),
            education_level=rare_values.get("education_level")
        )
        
        logger.info(f"Rare sampling: {condition}")
        return condition
    
    def _distribution_based_selection(self) -> GenerationCondition:
        """基于分布的选择（原有逻辑，保持90%权重）"""
        # 调用父类的 get_next_condition() 方法
        return self.get_next_condition()
    
    def _discriminator_guided_selection(self) -> Optional[GenerationCondition]:
        """
        判别器引导选择（降低到5%）
        
        只在质量明显不足时启用
        """
        if not self.generated_samples or not self.sample_scores:
            return None
        
        # 只分析最近100个样本（避免过度依赖历史）
        recent_samples = self.generated_samples[-100:]
        recent_scores = self.sample_scores[-100:]
        
        # 如果整体质量已经很高（>0.9），跳过判别器引导
        avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
        if avg_score > 0.9:
            logger.info(f"Quality sufficient ({avg_score:.3f}), skip discriminator guidance")
            return None
        
        # 找出低质量类别
        category_scores = defaultdict(list)
        for sample, score in zip(recent_samples, recent_scores):
            income = sample.get("income")
            age = sample.get("age", 0)
            age_range = "young" if age <= 30 else ("middle" if age <= 55 else "senior")
            
            if income:
                category_scores[f"income_{income}"].append(score)
            category_scores[f"age_{age_range}"].append(score)
        
        # 找出平均分最低的类别
        weak_categories = []
        for category, scores in category_scores.items():
            if len(scores) >= 5:  # 至少5个样本
                avg = sum(scores) / len(scores)
                weak_categories.append((category, avg))
        
        if not weak_categories:
            return None
        
        weak_categories.sort(key=lambda x: x[1])
        weakest = weak_categories[0][0]
        
        # 解析类别
        condition = GenerationCondition()
        if weakest.startswith("income_"):
            condition.income = weakest.split("_")[1]
        elif weakest.startswith("age_"):
            condition.age_range = weakest.split("_")[1]
        
        logger.info(f"Discriminator guidance: target weak category {weakest} (score={weak_categories[0][1]:.3f})")
        return condition
    
    def select_next_condition(self) -> GenerationCondition:
        """
        选择下一个生成条件（改进版）
        
        策略权重：
        - 90%: 分布探索（自适应校正）
        - 5%: 判别器引导
        - 5%: 稀有值采样
        
        依靠自适应缓存机制平衡分布，无需去重
        """
        # 更新多样性指标
        self._update_diversity_metrics()
        
        # 动态调整策略：如果多样性很低，增加稀有值采样
        if self.diversity_metrics.uniqueness_rate < 0.3:
            adjusted_rare_ratio = 0.15  # 提升到15%
            adjusted_disc_ratio = 0.05
        else:
            adjusted_rare_ratio = self.rare_sampling_ratio
            adjusted_disc_ratio = self.discriminator_ratio
        
        # 随机选择策略
        rand = random.random()
        
        if rand < adjusted_rare_ratio:
            # 稀有值采样
            condition = self._select_rare_values()
        elif rand < adjusted_rare_ratio + adjusted_disc_ratio:
            # 判别器引导
            condition = self._discriminator_guided_selection()
            if condition is None:
                condition = self._distribution_based_selection()
        else:
            # 分布探索（主要策略，配合自适应校正）
            condition = self._distribution_based_selection()
        
        # 更新字段频率统计
        if condition.income:
            self.field_value_counts["income"][condition.income] += 1
        if condition.age_range:
            self.field_value_counts["age_range"][condition.age_range] += 1
        if condition.education_level:
            self.field_value_counts["education_level"][condition.education_level] += 1
        
        return condition
    
    
    def add_sample(self, sample: Dict, score: float = None):
        """添加生成的样本（覆盖父类方法）"""
        # 更新父类统计
        self.update(sample)
        
        # 记录样本和分数
        self.generated_samples.append(sample)
        if score is not None:
            self.sample_scores.append(score)
        
        # 每100个样本报告一次多样性指标
        if len(self.generated_samples) % 100 == 0:
            self._update_diversity_metrics()
            logger.info(f"Diversity metrics at {len(self.generated_samples)} samples:")
            logger.info(f"  - Unique combinations: {self.diversity_metrics.unique_combinations}")
            logger.info(f"  - Uniqueness rate: {self.diversity_metrics.uniqueness_rate*100:.1f}%")
            logger.info(f"  - Coverage rate: {self.diversity_metrics.coverage_rate*100:.1f}%")
