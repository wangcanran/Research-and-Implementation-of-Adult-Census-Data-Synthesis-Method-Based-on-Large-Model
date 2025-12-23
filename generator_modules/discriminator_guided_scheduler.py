"""
Discriminator-Guided Scheduler
基于判别器反馈的调度器 - 主动学习策略
"""

import random
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict, Counter

from .task_spec import GenerationCondition


class DiscriminatorGuidedScheduler:
    """
    判别器引导的调度器
    
    核心思想：
    1. 生成样本 → 判别器评分
    2. 分析哪些类别的真实度评分低
    3. 下一轮优先生成该类别（主动学习）
    
    优势：
    - 自适应：自动发现生成质量低的类别
    - 主动学习：优先生成难样本
    - 质量驱动：不仅看分布，还看真实度
    """
    
    def __init__(self, discriminator, 
                 target_distribution: Optional[Dict] = None,
                 exploration_rate: float = 0.2):
        """
        Args:
            discriminator: 整体判别器（已训练）
            target_distribution: 目标分布（可选）
            exploration_rate: 探索率（0-1），控制基于分布vs基于判别器的比例
        """
        self.discriminator = discriminator
        self.target_distribution = target_distribution or {
            "income": {"<=50K": 0.76, ">50K": 0.24},
            "age_range": {"young": 0.25, "middle": 0.55, "senior": 0.20},
            "education_level": {"low": 0.30, "medium": 0.40, "high": 0.30}
        }
        
        self.exploration_rate = exploration_rate
        
        # 记录已生成样本及其判别器评分
        self.generated_samples = []
        self.sample_scores = []
        
        # 当前分布统计
        self.current_distribution = {
            "income": Counter(),
            "age_range": Counter(),
            "education_level": Counter()
        }
        
        self.iteration = 0
    
    def update(self, sample: Dict, score: Optional[float] = None):
        """
        更新已生成样本和评分
        
        Args:
            sample: 生成的样本
            score: 判别器评分（如果为None，会自动评估）
        """
        self.generated_samples.append(sample)
        
        # 获取判别器评分
        if score is None and self.discriminator and hasattr(self.discriminator, 'score'):
            score = self.discriminator.score(sample)
        
        if score is not None:
            self.sample_scores.append(score)
        
        # 更新分布统计
        self._update_distribution(sample)
    
    def _update_distribution(self, sample: Dict):
        """更新分布统计"""
        # Income
        income = sample.get("income")
        if income in ["<=50K", ">50K"]:
            self.current_distribution["income"][income] += 1
        
        # Age range
        try:
            age = int(sample.get("age", 0))
            if age <= 30:
                self.current_distribution["age_range"]["young"] += 1
            elif age <= 55:
                self.current_distribution["age_range"]["middle"] += 1
            else:
                self.current_distribution["age_range"]["senior"] += 1
        except:
            pass
        
        # Education level
        try:
            edu_num = int(sample.get("education.num", 9))
            if edu_num <= 8:
                self.current_distribution["education_level"]["low"] += 1
            elif edu_num <= 12:
                self.current_distribution["education_level"]["medium"] += 1
            else:
                self.current_distribution["education_level"]["high"] += 1
        except:
            pass
    
    def get_next_condition(self) -> GenerationCondition:
        """
        获取下一个生成条件
        
        策略（混合）：
        - exploration_rate % 基于目标分布（保证分布平衡）
        - (1-exploration_rate) % 基于判别器反馈（提升质量）
        """
        self.iteration += 1
        
        # 冷启动：前50个样本使用目标分布
        if len(self.generated_samples) < 50:
            return self._target_distribution_based()
        
        # 混合策略
        if random.random() < self.exploration_rate:
            return self._target_distribution_based()
        else:
            return self._discriminator_guided()
    
    def _target_distribution_based(self) -> GenerationCondition:
        """基于目标分布的调度（原始策略）"""
        # Income - 选择差距最大的
        income = self._select_by_gap("income", {
            "<=50K": "<=50K",
            ">50K": ">50K"
        })
        
        # Age range
        age_range = self._select_by_gap("age_range", {
            "young": "young",
            "middle": "middle",
            "senior": "senior"
        })
        
        # Education level
        education_level = self._select_by_gap("education_level", {
            "low": "low",
            "medium": "medium",
            "high": "high"
        })
        
        return GenerationCondition(
            income=income,
            age_range=age_range,
            education_level=education_level
        )
    
    def _discriminator_guided(self) -> GenerationCondition:
        """
        基于判别器反馈的调度（主动学习）
        
        策略：
        1. 分析最近100个样本各类别的平均真实度
        2. 选择真实度最低的类别
        3. 下一轮生成该类别
        """
        if not self.sample_scores:
            return self._target_distribution_based()
        
        # 分析最近100个样本
        recent_samples = self.generated_samples[-100:]
        recent_scores = self.sample_scores[-100:]
        
        # 计算各类别的平均真实度
        category_scores = self._analyze_category_scores(recent_samples, recent_scores)
        
        # 找出真实度最低的类别
        weak_income = self._find_weakest_category(category_scores.get("income", {}))
        weak_age = self._find_weakest_category(category_scores.get("age_range", {}))
        weak_edu = self._find_weakest_category(category_scores.get("education_level", {}))
        
        return GenerationCondition(
            income=weak_income,
            age_range=weak_age,
            education_level=weak_edu
        )
    
    def _analyze_category_scores(self, samples: List[Dict], scores: List[float]) -> Dict:
        """分析各类别的真实度评分"""
        category_scores = {
            "income": defaultdict(list),
            "age_range": defaultdict(list),
            "education_level": defaultdict(list)
        }
        
        for sample, score in zip(samples, scores):
            # Income
            income = sample.get("income")
            if income:
                category_scores["income"][income].append(score)
            
            # Age range
            try:
                age = int(sample.get("age", 0))
                if age <= 30:
                    category_scores["age_range"]["young"].append(score)
                elif age <= 55:
                    category_scores["age_range"]["middle"].append(score)
                else:
                    category_scores["age_range"]["senior"].append(score)
            except:
                pass
            
            # Education level
            try:
                edu_num = int(sample.get("education.num", 9))
                if edu_num <= 8:
                    category_scores["education_level"]["low"].append(score)
                elif edu_num <= 12:
                    category_scores["education_level"]["medium"].append(score)
                else:
                    category_scores["education_level"]["high"].append(score)
            except:
                pass
        
        # 计算平均分
        avg_scores = {}
        for dimension, scores_dict in category_scores.items():
            avg_scores[dimension] = {}
            for category, score_list in scores_dict.items():
                if score_list:
                    avg_scores[dimension][category] = np.mean(score_list)
        
        return avg_scores
    
    def _find_weakest_category(self, category_scores: Dict[str, float]) -> Optional[str]:
        """找出真实度最低的类别"""
        if not category_scores:
            return None
        
        # 返回评分最低的类别
        weakest = min(category_scores.items(), key=lambda x: x[1])
        return weakest[0]
    
    def _select_by_gap(self, dimension: str, mapping: Dict) -> Optional[str]:
        """根据分布差距选择类别（原始策略）"""
        target_dist = self.target_distribution.get(dimension, {})
        current_counts = self.current_distribution.get(dimension, Counter())
        
        total = sum(current_counts.values())
        if total == 0:
            if target_dist:
                return random.choice(list(mapping.values()))
            return None
        
        # 计算差距
        gaps = {}
        for category, target_prob in target_dist.items():
            current_prob = current_counts.get(category, 0) / total
            gaps[category] = target_prob - current_prob
        
        if gaps:
            max_gap_category = max(gaps.items(), key=lambda x: x[1])[0]
            return mapping.get(max_gap_category)
        
        return None
    
    def get_quality_report(self) -> Dict:
        """获取质量报告"""
        if not self.sample_scores:
            return {"message": "No samples scored yet"}
        
        recent_scores = self.sample_scores[-100:] if len(self.sample_scores) > 100 else self.sample_scores
        
        category_scores = self._analyze_category_scores(
            self.generated_samples[-len(recent_scores):],
            recent_scores
        )
        
        return {
            "total_samples": len(self.generated_samples),
            "mean_score": float(np.mean(self.sample_scores)),
            "recent_mean_score": float(np.mean(recent_scores)),
            "category_quality": {
                dimension: {
                    cat: float(score)
                    for cat, score in scores.items()
                }
                for dimension, scores in category_scores.items()
            },
            "distribution": self.get_current_distribution()
        }
    
    def get_current_distribution(self) -> Dict:
        """获取当前分布"""
        dist = {}
        for dimension, counts in self.current_distribution.items():
            total = sum(counts.values())
            if total > 0:
                dist[dimension] = {
                    cat: count / total
                    for cat, count in counts.items()
                }
            else:
                dist[dimension] = {}
        return dist


class HybridScheduler:
    """
    混合调度器：目标分布 + 判别器反馈
    
    简化版：80%基于目标分布，20%基于判别器反馈
    """
    
    def __init__(self, discriminator, target_distribution: Optional[Dict] = None):
        """
        Args:
            discriminator: 整体判别器
            target_distribution: 目标分布
        """
        self.discriminator_scheduler = DiscriminatorGuidedScheduler(
            discriminator=discriminator,
            target_distribution=target_distribution,
            exploration_rate=0.8  # 80%基于分布，20%基于判别器
        )
    
    def update(self, sample: Dict, score: Optional[float] = None):
        """更新"""
        self.discriminator_scheduler.update(sample, score)
    
    def get_next_condition(self) -> GenerationCondition:
        """获取下一个条件"""
        return self.discriminator_scheduler.get_next_condition()
    
    def get_quality_report(self) -> Dict:
        """获取质量报告"""
        return self.discriminator_scheduler.get_quality_report()
    
    def get_current_distribution(self) -> Dict:
        """获取当前分布"""
        return self.discriminator_scheduler.get_current_distribution()
