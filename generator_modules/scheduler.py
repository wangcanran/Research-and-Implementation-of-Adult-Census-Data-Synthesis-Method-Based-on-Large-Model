"""
Dataset-Wise Scheduler Module
从 data_generator.py 提取，修改为 Adult Census
"""

from typing import Dict, Optional
from collections import Counter
from .task_spec import GenerationCondition


class DatasetWiseScheduler:
    """
    数据集级调度器（照搬data_generator.py）
    
    功能：
    1. 基于目标分布调度生成条件
    2. 跟踪已生成样本统计
    3. 选择与目标分布差距最大的类别优先生成
    """
    
    def __init__(self, target_distribution: Optional[Dict] = None):
        """
        Args:
            target_distribution: 目标分布（修改为Adult Census）
        """
        # 默认目标分布（基于Adult Census真实分布）
        self.target_distribution = target_distribution or {
            "income": {
                "<=50K": 0.76,  # 76% low income
                ">50K": 0.24    # 24% high income
            },
            "age_range": {
                "young": 0.25,   # 17-30
                "middle": 0.55,  # 31-55
                "senior": 0.20   # 56-90
            },
            "education_level": {
                "low": 0.30,      # 1-8 years
                "medium": 0.40,   # 9-12 years
                "high": 0.30      # 13-16 years
            }
        }
        
        # 当前生成统计
        self.generated_stats = {
            "income": Counter(),
            "age_range": Counter(),
            "education_level": Counter()
        }
    
    def get_next_condition(self) -> GenerationCondition:
        """
        获取下一个生成条件（照搬data_generator.py的策略）
        
        策略：选择与目标分布差距最大的类别
        """
        condition = GenerationCondition()
        
        # Income - 选择差距最大的
        income = self._select_by_gap("income", {
            "<=50K": "<=50K",
            ">50K": ">50K"
        })
        condition.income = income
        
        # Age range - 选择差距最大的
        age_range = self._select_by_gap("age_range", {
            "young": "young",
            "middle": "middle",
            "senior": "senior"
        })
        condition.age_range = age_range
        
        # Education level - 选择差距最大的
        education_level = self._select_by_gap("education_level", {
            "low": "low",
            "medium": "medium",
            "high": "high"
        })
        condition.education_level = education_level
        
        return condition
    
    def _select_by_gap(self, category: str, mapping: Dict) -> Optional[str]:
        """选择与目标分布差距最大的类别（照搬data_generator.py）"""
        total = sum(self.generated_stats[category].values()) or 1
        
        gaps = {}
        for key, target_prob in self.target_distribution[category].items():
            current_prob = self.generated_stats[category].get(key, 0) / total
            gaps[key] = target_prob - current_prob
        
        # 选择差距最大的
        max_gap_key = max(gaps, key=gaps.get)
        return mapping.get(max_gap_key)
    
    def update(self, sample: Dict):
        """更新统计（照搬data_generator.py）"""
        # Income
        income = sample.get("income")
        if income in ["<=50K", ">50K"]:
            self.generated_stats["income"][income] += 1
        
        # Age range
        try:
            age = int(sample.get("age", 0))
            if age <= 30:
                age_range = "young"
            elif age <= 55:
                age_range = "middle"
            else:
                age_range = "senior"
            self.generated_stats["age_range"][age_range] += 1
        except:
            self.generated_stats["age_range"]["middle"] += 1
        
        # Education level
        try:
            edu_num = int(sample.get("education.num", 9))
            if edu_num <= 8:
                edu_level = "low"
            elif edu_num <= 12:
                edu_level = "medium"
            else:
                edu_level = "high"
            self.generated_stats["education_level"][edu_level] += 1
        except:
            self.generated_stats["education_level"]["medium"] += 1
