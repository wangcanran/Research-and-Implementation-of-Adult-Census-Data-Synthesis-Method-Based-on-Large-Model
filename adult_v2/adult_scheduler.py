"""
Adult Census Data - Dataset-Wise Scheduler
数据集调度器 - 基于目标分布的条件调度
"""

from typing import Dict, Optional
from collections import Counter
from .adult_task_spec import GenerationCondition


class AdultDatasetWiseScheduler:
    """
    数据集调度器
    
    功能：
    1. 根据目标分布生成GenerationCondition
    2. 跟踪已生成数据的分布
    3. 基于差距调度下一个生成条件（Metric Based Scheduling）
    """
    
    def __init__(self, target_distribution: Optional[Dict] = None):
        """
        Args:
            target_distribution: 目标分布
                {
                    "age_range": {"young": 0.3, "middle": 0.5, "senior": 0.2},
                    "education_level": {"low": 0.3, "medium": 0.5, "high": 0.2},
                    "income": {"<=50K": 0.76, ">50K": 0.24},
                    "gender": {"Male": 0.67, "Female": 0.33}
                }
        """
        if target_distribution is None:
            # 默认目标分布（接近真实Adult数据）
            self.target_distribution = {
                "age_range": {"young": 0.25, "middle": 0.50, "senior": 0.25},
                "education_level": {"low": 0.30, "medium": 0.50, "high": 0.20},
                "income": {"<=50K": 0.76, ">50K": 0.24},
                "gender": {"Male": 0.67, "Female": 0.33}
            }
        else:
            self.target_distribution = target_distribution
        
        # 跟踪当前已生成的分布
        self.current_distribution = {
            "age_range": Counter(),
            "education_level": Counter(),
            "income": Counter(),
            "gender": Counter()
        }
    
    def get_next_condition(self) -> GenerationCondition:
        """
        基于指标的调度 (Metric Based Scheduling)
        
        策略：
        1. 计算当前分布与目标分布的差距
        2. 选择差距最大的维度优先填补
        3. 生成对应的GenerationCondition
        """
        # 为每个维度选择需要补充的类别
        age_range = self._select_by_gap("age_range", {
            "young": "young",
            "middle": "middle",
            "senior": "senior"
        })
        
        education_level = self._select_by_gap("education_level", {
            "low": "low",
            "medium": "medium",
            "high": "high"
        })
        
        income_class = self._select_by_gap("income", {
            "<=50K": "<=50K",
            ">50K": ">50K"
        })
        
        gender = self._select_by_gap("gender", {
            "Male": "Male",
            "Female": "Female"
        })
        
        return GenerationCondition(
            age_range=age_range,
            education_level=education_level,
            income_class=income_class,
            gender=gender
        )
    
    def _select_by_gap(self, dimension: str, mapping: Dict) -> Optional[str]:
        """
        根据差距选择类别
        
        计算公式：gap = target_ratio - current_ratio
        选择gap最大的类别
        """
        target_dist = self.target_distribution.get(dimension, {})
        current_counts = self.current_distribution.get(dimension, Counter())
        
        # 计算总数
        total_count = sum(current_counts.values())
        
        if total_count == 0:
            # 如果还没生成，随机选择一个有目标的类别
            import random
            if target_dist:
                return random.choice(list(mapping.values()))
            else:
                return None
        
        # 计算每个类别的差距
        gaps = {}
        for category, target_ratio in target_dist.items():
            current_ratio = current_counts.get(category, 0) / total_count
            gap = target_ratio - current_ratio
            gaps[category] = gap
        
        # 选择差距最大的
        if gaps:
            max_gap_category = max(gaps.items(), key=lambda x: x[1])[0]
            return mapping.get(max_gap_category)
        else:
            return None
    
    def update(self, sample: Dict):
        """
        更新当前分布统计
        
        Args:
            sample: 已生成的样本
        """
        # 更新age_range
        age = sample.get("age")
        if age:
            try:
                age_val = int(age)
                if 17 <= age_val <= 30:
                    self.current_distribution["age_range"]["young"] += 1
                elif 31 <= age_val <= 55:
                    self.current_distribution["age_range"]["middle"] += 1
                elif 56 <= age_val <= 90:
                    self.current_distribution["age_range"]["senior"] += 1
            except:
                pass
        
        # 更新education_level
        education_num = sample.get("education.num")
        if education_num:
            try:
                edu_val = int(education_num)
                if edu_val <= 8:
                    self.current_distribution["education_level"]["low"] += 1
                elif 9 <= edu_val <= 12:
                    self.current_distribution["education_level"]["medium"] += 1
                elif edu_val >= 13:
                    self.current_distribution["education_level"]["high"] += 1
            except:
                pass
        
        # 更新income
        income = sample.get("income")
        if income:
            self.current_distribution["income"][income] += 1
        
        # 更新gender
        gender = sample.get("sex")
        if gender:
            self.current_distribution["gender"][gender] += 1
    
    def get_current_stats(self) -> Dict:
        """获取当前分布的统计信息"""
        stats = {}
        
        for dimension, counts in self.current_distribution.items():
            total = sum(counts.values())
            if total > 0:
                stats[dimension] = {
                    category: count / total
                    for category, count in counts.items()
                }
            else:
                stats[dimension] = {}
        
        return stats
    
    def get_distribution_gap(self) -> Dict:
        """计算当前分布与目标分布的差距"""
        gaps = {}
        current_stats = self.get_current_stats()
        
        for dimension, target_dist in self.target_distribution.items():
            current_dist = current_stats.get(dimension, {})
            dimension_gaps = {}
            
            for category, target_ratio in target_dist.items():
                current_ratio = current_dist.get(category, 0)
                gap = abs(target_ratio - current_ratio)
                dimension_gaps[category] = gap
            
            gaps[dimension] = dimension_gaps
        
        return gaps
