"""
Adult Census Data - Demonstration Manager
示例管理器 - 启发式高质量样本选择
"""

import random
from typing import List, Dict, Optional
from .adult_task_spec import (
    GenerationCondition, EDUCATION_MAPPING, 
    validate_education_mapping, validate_marital_relationship
)


class AdultDemonstrationManager:
    """
    示例管理器
    
    功能：
    1. 加载真实Adult数据作为示例库
    2. 启发式选择高质量示例
    3. 基于条件选择相似示例
    """
    
    def __init__(self, use_heuristic: bool = True):
        """
        Args:
            use_heuristic: 是否使用启发式高质量选择
        """
        self.demonstrations = []  # 预定义示例
        self.real_samples = []    # 从真实数据加载的样本
        self.use_heuristic = use_heuristic
        self.sample_quality_cache = {}  # 缓存样本质量评分
    
    def load_samples_from_file(self, file_path: str):
        """从CSV文件加载真实样本"""
        import pandas as pd
        
        try:
            df = pd.read_csv(file_path)
            # 转换为字典列表
            self.real_samples = df.to_dict('records')
            print(f"  [DemoManager] 加载 {len(self.real_samples)} 个真实样本")
        except Exception as e:
            print(f"  [DemoManager] 加载样本失败: {e}")
    
    def select_demonstrations(self, condition: GenerationCondition, k: int = 2,
                             use_multi_candidate: bool = True) -> List[Dict]:
        """
        示例选择策略
        
        策略：
        1. 预定义示例 - 手工构造的高质量示例
        2. 启发式选择 - 从真实样本中选择高质量且相似的
        3. 随机选择 - 从真实样本中随机选择
        4. 多候选验证 - 生成多个候选并选择最一致的
        
        Args:
            condition: 生成条件
            k: 返回的示例数量
            use_multi_candidate: 是否使用多候选验证
        
        Returns:
            选中的示例列表
        """
        if self.use_heuristic and self.real_samples:
            if use_multi_candidate:
                return self._multi_candidate_select(condition, k)
            else:
                return self._heuristic_select(condition, k)
        elif self.real_samples:
            return random.sample(self.real_samples, min(k, len(self.real_samples)))
        else:
            return self.demonstrations[:k]
    
    def _heuristic_select(self, condition: GenerationCondition, k: int) -> List[Dict]:
        """
        启发式高质量样本选择
        
        核心思路：
        1. 计算每个样本的质量分数（完整性、一致性、逻辑合规性）
        2. 计算每个样本与条件的相似度
        3. 综合质量和相似度排序
        """
        if not self.real_samples:
            return []
        
        scored_samples = []
        
        for sample in self.real_samples:
            # 1. 质量分数（0-1）
            quality = self._calculate_quality_score(sample)
            
            # 2. 相似度分数（0-1）
            similarity = self._calculate_similarity(sample, condition)
            
            # 3. 综合分数：质量60% + 相似度40%
            combined_score = 0.6 * quality + 0.4 * similarity
            
            scored_samples.append((sample, combined_score, quality, similarity))
        
        # 按综合分数降序排序
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top-k
        selected = [s[0] for s in scored_samples[:k]]
        
        return selected
    
    def _calculate_quality_score(self, sample: Dict) -> float:
        """
        样本质量评分
        
        评估标准：
        1. 字段完整性（30%）
        2. 教育-年限一致性（25%）
        3. 婚姻-关系-性别一致性（25%）
        4. 数值合理性（20%）
        """
        # 使用缓存
        sample_id = str(sample.get("fnlwgt", random.randint(1, 1000000)))
        if sample_id in self.sample_quality_cache:
            return self.sample_quality_cache[sample_id]
        
        score = 1.0
        
        # 1. 字段完整性检查（占30%）
        required_fields = ["age", "education", "occupation", "marital.status", 
                          "relationship", "sex", "income"]
        missing = sum(1 for f in required_fields if not sample.get(f))
        completeness = 1.0 - (missing / len(required_fields))
        score *= (0.3 * completeness + 0.7)
        
        # 2. 教育-年限一致性（占25%）
        try:
            education = sample.get("education", "")
            education_num = int(sample.get("education.num", 0))
            if validate_education_mapping(education, education_num):
                score *= 1.0
            else:
                score *= 0.5
        except:
            score *= 0.6
        
        # 3. 婚姻-关系-性别一致性（占25%）
        try:
            marital = sample.get("marital.status", "")
            sex = sample.get("sex", "")
            relationship = sample.get("relationship", "")
            
            if validate_marital_relationship(marital, sex, relationship):
                score *= 1.0
            else:
                score *= 0.4
        except:
            score *= 0.6
        
        # 4. 数值合理性（占20%）
        try:
            age = int(sample.get("age", 0))
            hours = int(sample.get("hours.per.week", 0))
            
            if 17 <= age <= 90:
                score *= 1.0
            else:
                score *= 0.5
            
            if 1 <= hours <= 99:
                score *= 1.0
            else:
                score *= 0.5
        except:
            score *= 0.7
        
        # 缓存结果
        self.sample_quality_cache[sample_id] = max(0.0, min(1.0, score))
        return self.sample_quality_cache[sample_id]
    
    def _calculate_similarity(self, sample: Dict, condition: GenerationCondition) -> float:
        """计算样本与生成条件的相似度"""
        similarity = 0.0
        matches = 0
        total = 0
        
        # Age Range Matching
        if condition.age_range:
            total += 1
            try:
                age = int(sample.get("age", 0))
                if condition.age_range == "young" and 17 <= age <= 30:
                    matches += 1
                elif condition.age_range == "middle" and 31 <= age <= 55:
                    matches += 1
                elif condition.age_range == "senior" and 56 <= age <= 90:
                    matches += 1
            except:
                pass
        
        # Education Level Matching
        if condition.education_level:
            total += 1
            try:
                edu_num = int(sample.get("education.num", 0))
                if condition.education_level == "low" and edu_num <= 8:
                    matches += 1
                elif condition.education_level == "medium" and 9 <= edu_num <= 12:
                    matches += 1
                elif condition.education_level == "high" and edu_num >= 13:
                    matches += 1
            except:
                pass
        
        # Gender Matching
        if condition.gender:
            total += 1
            if sample.get("sex") == condition.gender:
                matches += 1
        
        # Income Matching
        if condition.income_class:
            total += 1
            if sample.get("income") == condition.income_class:
                matches += 1
        
        # Marital Status Matching
        if condition.marital_status:
            total += 1
            marital = sample.get("marital.status", "")
            if condition.marital_status == "married" and "Married" in marital:
                matches += 1
            elif condition.marital_status == "single" and marital == "Never-married":
                matches += 1
            elif condition.marital_status == "divorced" and marital in ["Divorced", "Separated", "Widowed"]:
                matches += 1
        
        # 计算相似度
        if total > 0:
            similarity = matches / total
        else:
            similarity = 0.5  # 无条件时返回中等相似度
        
        return similarity
    
    def _multi_candidate_select(self, condition: GenerationCondition, k: int, 
                               n_runs: int = 3) -> List[Dict]:
        """
        多候选相互验证选择策略
        
        思路：
        1. 运行n_runs次启发式选择
        2. 统计每个样本被选中的次数
        3. 选择最频繁被选中的k个样本
        """
        if not self.real_samples:
            return []
        
        # 记录每个样本被选中的次数
        selection_counts = {}
        
        for _ in range(n_runs):
            selected = self._heuristic_select(condition, k)
            for sample in selected:
                sample_id = str(sample.get("fnlwgt", id(sample)))
                if sample_id not in selection_counts:
                    selection_counts[sample_id] = {"sample": sample, "count": 0}
                selection_counts[sample_id]["count"] += 1
        
        # 按被选中次数排序
        sorted_samples = sorted(
            selection_counts.values(),
            key=lambda x: x["count"],
            reverse=True
        )
        
        # 返回top-k
        return [item["sample"] for item in sorted_samples[:k]]
