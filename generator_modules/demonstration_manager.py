"""
Demonstration Manager Module
从 data_generator.py 提取，修改为 Adult Census
"""

import random
from typing import List, Dict, Optional
from .task_spec import GenerationCondition

class DemonstrationManager:
    """
    示例管理器（照搬data_generator.py策略）
    
    功能：
    1. 从真实数据中选择示例（基于SuperGen的启发式选择）
    2. 格式化示例为Prompt
    3. 支持基于条件的示例筛选
    """
    
    def __init__(self, use_heuristic: bool = True):
        """
        Args:
            use_heuristic: 是否使用启发式选择（基于SuperGen）
        """
        self.real_samples = []
        self.use_heuristic = use_heuristic
        self.sample_quality_cache = {}  # 缓存样本质量评分
    
    def load_samples(self, samples: List[Dict]):
        """加载真实样本"""
        self.real_samples = samples
        self.sample_quality_cache.clear()
    
    def select_demonstrations(self, k: int = 3, condition: Optional[GenerationCondition] = None) -> List[Dict]:
        """
        选择示例（照搬data_generator.py的SuperGen策略）
        
        Args:
            k: 选择数量
            condition: 生成条件
        
        Returns:
            选择的示例列表
        """
        if not self.real_samples:
            return []
        
        if not self.use_heuristic or condition is None:
            # 随机选择
            return random.sample(self.real_samples, min(k, len(self.real_samples)))
        
        # SuperGen启发式选择（照搬）
        return self._supergen_select(k, condition)
    
    def _supergen_select(self, k: int, condition: GenerationCondition) -> List[Dict]:
        """
        SuperGen启发式选择（照搬data_generator.py）
        
        策略：
        1. 质量评分
        2. 相似度评分
        3. 不确定性评分
        4. 多样性保证
        """
        # Step 1: 计算每个样本的综合得分
        scored_samples = []
        for sample in self.real_samples:
            quality = self._calculate_quality_score(sample)
            similarity = self._calculate_similarity(sample, condition)
            uncertainty = self._calculate_uncertainty(sample)
            
            # 综合得分（照搬data_generator.py的权重）
            score = quality * 0.4 + similarity * 0.4 + uncertainty * 0.2
            scored_samples.append((sample, score, uncertainty))
        
        # Step 2: 按综合得分排序
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: 不确定性过滤
        filtered_samples = [(s, score) for s, score, unc in scored_samples if unc > 0.2]
        
        if not filtered_samples:
            filtered_samples = [(s, score) for s, score, _ in scored_samples]
        
        # Step 4: 从top-2k中选择k个多样化样本
        candidate_pool = filtered_samples[:min(k * 2, len(filtered_samples))]
        selected = self._diverse_sampling([s for s, _ in candidate_pool], k)
        
        return selected
    
    def _calculate_quality_score(self, sample: Dict) -> float:
        """样本质量评分（修改为Adult Census）"""
        score = 1.0
        
        # 1. 字段完整性（30%）
        required_fields = ["age", "education", "occupation", "hours.per.week", "income"]
        missing = sum(1 for f in required_fields if not sample.get(f) or sample.get(f) == '?')
        completeness = 1.0 - (missing / len(required_fields))
        score *= (0.3 * completeness + 0.7)
        
        # 2. 逻辑一致性（40%）
        # 年龄-婚姻
        try:
            age = int(sample.get("age", 0))
            marital = sample.get("marital.status", "")
            if age < 18 and marital in ["Married-civ-spouse"]:
                score *= 0.3
            elif age < 25 and marital == "Married-civ-spouse":
                score *= 0.8
        except:
            score *= 0.9
        
        # 婚姻-关系
        marital = sample.get("marital.status", "")
        relationship = sample.get("relationship", "")
        if marital == "Married-civ-spouse" and relationship in ["Husband", "Wife"]:
            score *= 1.0
        elif marital == "Married-civ-spouse":
            score *= 0.6
        
        # 3. 合理性（30%）
        # 教育-收入
        try:
            edu_num = int(sample.get("education.num", 9))
            income = sample.get("income", "<=50K")
            
            if edu_num >= 13 and income == ">50K":
                score *= 1.0
            elif edu_num <= 9 and income == "<=50K":
                score *= 1.0
            else:
                score *= 0.9
        except:
            score *= 0.9
        
        return max(0.0, min(1.0, score))
    
    def _calculate_similarity(self, sample: Dict, condition: GenerationCondition) -> float:
        """计算样本与条件的相似度（修改为Adult Census）"""
        similarity = 0.0
        
        # Income匹配（40%）
        if condition.income:
            if sample.get("income") == condition.income:
                similarity += 0.4
        
        # Age range匹配（30%）
        if condition.age_range:
            try:
                age = int(sample.get("age", 0))
                age_ranges = {"young": (17, 30), "middle": (31, 55), "senior": (56, 90)}
                if condition.age_range in age_ranges:
                    min_age, max_age = age_ranges[condition.age_range]
                    if min_age <= age <= max_age:
                        similarity += 0.3
            except:
                pass
        
        # Education level匹配（30%）
        if condition.education_level:
            try:
                edu_num = int(sample.get("education.num", 9))
                edu_ranges = {"low": (1, 8), "medium": (9, 12), "high": (13, 16)}
                if condition.education_level in edu_ranges:
                    min_edu, max_edu = edu_ranges[condition.education_level]
                    if min_edu <= edu_num <= max_edu:
                        similarity += 0.3
            except:
                pass
        
        return similarity
    
    def _calculate_uncertainty(self, sample: Dict) -> float:
        """计算样本的不确定性/多样性（照搬data_generator.py）"""
        # 简单实现：基于字段值的多样性
        uncertainty = 0.5  # 默认中等不确定性
        
        # 如果有少见值，增加不确定性
        income = sample.get("income")
        if income == ">50K":  # 高收入较少见
            uncertainty += 0.2
        
        try:
            edu_num = int(sample.get("education.num", 9))
            if edu_num >= 14:  # 高学历较少见
                uncertainty += 0.2
        except:
            pass
        
        return min(1.0, uncertainty)
    
    def _diverse_sampling(self, candidates: List[Dict], k: int) -> List[Dict]:
        """多样性采样（照搬data_generator.py）"""
        if len(candidates) <= k:
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        # 贪心选择：每次选择与已选样本最不相似的
        if remaining:
            selected.append(remaining.pop(0))  # 先选质量最高的
        
        while len(selected) < k and remaining:
            max_min_dist = -1
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                min_dist = min(self._sample_distance(candidate, s) for s in selected)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _sample_distance(self, s1: Dict, s2: Dict) -> float:
        """计算两个样本的距离（修改为Adult Census）"""
        distance = 0.0
        
        # Age距离
        try:
            age1 = int(s1.get("age", 0))
            age2 = int(s2.get("age", 0))
            distance += abs(age1 - age2) / 90.0
        except:
            distance += 0.5
        
        # Education距离
        try:
            edu1 = int(s1.get("education.num", 9))
            edu2 = int(s2.get("education.num", 9))
            distance += abs(edu1 - edu2) / 16.0
        except:
            distance += 0.5
        
        # Income不同
        if s1.get("income") != s2.get("income"):
            distance += 1.0
        
        # Occupation不同
        if s1.get("occupation") != s2.get("occupation"):
            distance += 0.5
        
        return distance / 3.0  # 归一化
    
    def format_demonstrations(self, demos: List[Dict]) -> str:
        """格式化示例为prompt（照搬data_generator.py）"""
        if not demos:
            return ""
        
        formatted = "Here are some example census records:\n\n"
        for i, demo in enumerate(demos, 1):
            formatted += f"Example {i}:\n"
            formatted += "{\n"
            for key, value in demo.items():
                if not key.startswith("_"):  # 跳过内部字段
                    formatted += f'  "{key}": "{value}",\n'
            formatted = formatted.rstrip(',\n') + '\n'
            formatted += "}\n\n"
        
        return formatted
