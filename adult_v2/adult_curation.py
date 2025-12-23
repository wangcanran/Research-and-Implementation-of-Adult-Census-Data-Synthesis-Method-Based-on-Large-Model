"""
Adult Census Data - Curation Stage
策展阶段 - 样本过滤、重加权、增强
"""

import random
from typing import List, Dict, Tuple
from .adult_task_spec import validate_sample


class AdultSampleFilter:
    """
    样本过滤器
    
    功能：
    1. 过滤格式错误的样本
    2. 过滤逻辑不一致的样本
    3. 过滤质量过低的样本
    """
    
    def __init__(self, strict: bool = True):
        """
        Args:
            strict: 是否使用严格模式（严格模式会过滤更多样本）
        """
        self.strict = strict
    
    def filter_batch(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        批量过滤样本
        
        Returns:
            (passed_samples, failed_samples)
        """
        passed = []
        failed = []
        
        for sample in samples:
            is_valid, errors = self.filter_single(sample)
            if is_valid:
                passed.append(sample)
            else:
                failed.append({"sample": sample, "errors": errors})
        
        return passed, failed
    
    def filter_single(self, sample: Dict) -> Tuple[bool, List[str]]:
        """
        过滤单个样本
        
        Returns:
            (is_valid, error_messages)
        """
        # 使用task_spec中的验证函数
        is_valid, errors = validate_sample(sample)
        
        if not self.strict:
            # 非严格模式：只要没有致命错误就通过
            fatal_errors = [e for e in errors if "Missing field" in e or "mapping error" in e]
            return len(fatal_errors) == 0, fatal_errors
        else:
            # 严格模式：所有错误都要修复
            return is_valid, errors


class AdultSampleReweighter:
    """
    样本重加权器（SunGen双循环）
    
    功能：
    1. 根据样本质量分配权重
    2. 根据样本稀有度分配权重
    3. 支持基于目标分布的重加权
    """
    
    def __init__(self):
        self.weight_cache = {}
    
    def reweight_by_quality(self, samples: List[Dict]) -> Dict[int, float]:
        """
        基于质量的重加权
        
        策略：
        - 高质量样本权重高
        - 低质量样本权重低
        """
        weights = {}
        
        for idx, sample in enumerate(samples):
            # 计算质量分数（简化版）
            quality = self._calculate_quality(sample)
            weights[idx] = quality
        
        # 归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def reweight_by_rarity(self, samples: List[Dict]) -> Dict[int, float]:
        """
        基于稀有度的重加权
        
        策略：
        - 稀有组合权重高（避免过度采样常见组合）
        - 常见组合权重低
        """
        from collections import Counter
        
        # 统计组合频率
        combination_counts = Counter()
        sample_combinations = {}
        
        for idx, sample in enumerate(samples):
            # 创建组合key
            combination = self._get_combination_key(sample)
            combination_counts[combination] += 1
            sample_combinations[idx] = combination
        
        # 计算权重（频率的倒数）
        weights = {}
        total_samples = len(samples)
        
        for idx, combination in sample_combinations.items():
            frequency = combination_counts[combination] / total_samples
            # 权重 = 1 / 频率（稀有的权重高）
            weights[idx] = 1.0 / (frequency + 0.01)  # +0.01避免除零
        
        # 归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_quality(self, sample: Dict) -> float:
        """计算样本质量分数"""
        is_valid, errors = validate_sample(sample)
        
        # 基础分数
        score = 1.0 if is_valid else 0.5
        
        # 字段完整性
        required_fields = ["age", "education", "occupation", "income"]
        completeness = sum(1 for f in required_fields if sample.get(f)) / len(required_fields)
        score *= completeness
        
        return score
    
    def _get_combination_key(self, sample: Dict) -> str:
        """获取样本的组合key（用于稀有度计算）"""
        # 使用关键字段的组合
        age_range = "young" if sample.get("age", 0) < 30 else ("middle" if sample.get("age", 0) < 55 else "senior")
        education = sample.get("education", "")[:3]  # 前3个字符
        income = sample.get("income", "")
        
        return f"{age_range}_{education}_{income}"


class AdultLabelEnhancer:
    """
    标签增强器
    
    功能：
    1. 标签平滑（防止过拟合）
    2. 硬标签转软标签
    3. 基于置信度的标签调整
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: 平滑系数（0-1），越大平滑程度越高
        """
        self.smoothing = smoothing
    
    def enhance_batch(self, samples: List[Dict]) -> List[Dict]:
        """
        批量增强标签
        
        对于Adult数据，主要增强income标签
        """
        enhanced = []
        
        for sample in samples:
            enhanced_sample = sample.copy()
            
            # 对income进行标签平滑
            if "income" in sample:
                # 添加置信度字段（可选）
                income = sample["income"]
                if income == ">50K":
                    enhanced_sample["income_confidence"] = 1.0 - self.smoothing
                else:
                    enhanced_sample["income_confidence"] = 1.0 - self.smoothing
            
            enhanced.append(enhanced_sample)
        
        return enhanced


class AdultAuxiliaryModelEnhancer:
    """
    辅助模型增强器
    
    功能：
    1. 使用判别式模型增强样本（如果有训练好的模型）
    2. 预测缺失字段
    3. 修正不一致字段
    
    注：这是一个占位实现，实际需要训练的辅助模型
    """
    
    def __init__(self, use_discriminative: bool = True):
        """
        Args:
            use_discriminative: 是否使用判别式模型
        """
        self.use_discriminative = use_discriminative
        self.model = None  # 占位，实际需要加载训练好的模型
    
    def enhance_batch(self, samples: List[Dict]) -> List[Dict]:
        """
        批量增强样本
        
        目前是占位实现，直接返回原样本
        实际应该：
        1. 使用分类器预测income
        2. 使用模型修正不一致的字段
        3. 填充缺失字段
        """
        # 占位实现：直接返回
        return samples
    
    def predict_missing_fields(self, sample: Dict) -> Dict:
        """预测并填充缺失字段（占位）"""
        return sample
    
    def correct_inconsistencies(self, sample: Dict) -> Dict:
        """修正字段不一致（占位）"""
        return sample


# ============================================================================
#                      Curation Pipeline
# ============================================================================

class AdultCurationPipeline:
    """
    策展流程管道
    
    完整流程：Filter → Reweight → Label Enhance → Auxiliary Enhance
    """
    
    def __init__(self, use_reweighting: bool = True, use_enhancement: bool = True):
        """
        Args:
            use_reweighting: 是否使用重加权
            use_enhancement: 是否使用增强
        """
        self.filter = AdultSampleFilter(strict=True)
        self.reweighter = AdultSampleReweighter()
        self.label_enhancer = AdultLabelEnhancer(smoothing=0.1)
        self.auxiliary_enhancer = AdultAuxiliaryModelEnhancer(use_discriminative=True)
        
        self.use_reweighting = use_reweighting
        self.use_enhancement = use_enhancement
    
    def curate(self, samples: List[Dict]) -> Dict:
        """
        执行完整的策展流程
        
        Returns:
            {
                "curated_samples": List[Dict],
                "sample_weights": Dict[int, float],
                "filter_stats": Dict,
                "enhancement_stats": Dict
            }
        """
        result = {
            "original_count": len(samples),
            "curated_samples": [],
            "sample_weights": None,
            "filter_stats": {},
            "enhancement_stats": {}
        }
        
        # 1. Filter
        passed_samples, failed_samples = self.filter.filter_batch(samples)
        result["filter_stats"] = {
            "passed": len(passed_samples),
            "failed": len(failed_samples),
            "pass_rate": len(passed_samples) / len(samples) if samples else 0
        }
        
        if not passed_samples:
            return result
        
        # 2. Reweight (可选)
        if self.use_reweighting:
            quality_weights = self.reweighter.reweight_by_quality(passed_samples)
            rarity_weights = self.reweighter.reweight_by_rarity(passed_samples)
            
            # 组合权重：质量70% + 稀有度30%
            combined_weights = {}
            for idx in quality_weights:
                combined_weights[idx] = 0.7 * quality_weights[idx] + 0.3 * rarity_weights.get(idx, 0)
            
            result["sample_weights"] = combined_weights
        
        # 3. Label Enhancement (可选)
        if self.use_enhancement:
            enhanced_samples = self.label_enhancer.enhance_batch(passed_samples)
            result["enhancement_stats"]["label_enhanced"] = len(enhanced_samples)
        else:
            enhanced_samples = passed_samples
        
        # 4. Auxiliary Model Enhancement (可选)
        if self.use_enhancement:
            final_samples = self.auxiliary_enhancer.enhance_batch(enhanced_samples)
            result["enhancement_stats"]["auxiliary_enhanced"] = len(final_samples)
        else:
            final_samples = enhanced_samples
        
        result["curated_samples"] = final_samples
        result["final_count"] = len(final_samples)
        
        return result
