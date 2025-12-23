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
    
    支持两种模式：
    1. 字段级预测器：预测缺失字段、修正不一致
    2. 整体判别器：评估真实度、过滤低质量样本（推荐）
    """
    
    def __init__(self, model_dir: str = None, use_discriminative: bool = True,
                 use_holistic: bool = False, holistic_threshold: float = 0.5):
        """
        Args:
            model_dir: 字段级预测器模型目录
            use_discriminative: 是否使用字段级判别式模型
            use_holistic: 是否使用整体判别器（推荐）
            holistic_threshold: 整体判别器的真实度阈值
        """
        self.use_discriminative = use_discriminative
        self.use_holistic = use_holistic
        self.holistic_threshold = holistic_threshold
        self.models = None
        self.holistic_discriminator = None
        
        # 加载整体判别器（优先，推荐）
        if use_holistic:
            try:
                import os
                holistic_dir = "adult_v2/trained_holistic_discriminator"
                if os.path.exists(os.path.join(holistic_dir, 'holistic_discriminator.pkl')):
                    from .adult_holistic_discriminator import AdultHolisticDiscriminator
                    self.holistic_discriminator = AdultHolisticDiscriminator()
                    self.holistic_discriminator.load_model(holistic_dir)
                    print(f"  [整体判别器] 已加载，阈值={holistic_threshold}")
                else:
                    print(f"  [警告] 整体判别器未找到: {holistic_dir}")
                    print(f"  运行: python train_holistic_discriminator.py")
            except Exception as e:
                print(f"  [警告] 无法加载整体判别器: {e}")
        
        # 加载字段级预测器（可选，用于填充缺失字段）
        if use_discriminative and model_dir:
            try:
                import os
                # 优先尝试加载深度学习版本
                if os.path.exists(os.path.join(model_dir, 'model_income.pt')):
                    from .adult_discriminative_models_dl import AdultDiscriminativeModelsDL
                    self.models = AdultDiscriminativeModelsDL()
                    self.models.load_models(model_dir)
                # 尝试加载ML版本（多算法）
                elif os.path.exists(os.path.join(model_dir, 'models_ml.pkl')):
                    from .adult_discriminative_models_ml import AdultDiscriminativeModelsML
                    self.models = AdultDiscriminativeModelsML()
                    self.models.load_models(model_dir)
                # 回退到基础版本
                else:
                    from .adult_discriminative_models import AdultDiscriminativeModels
                    self.models = AdultDiscriminativeModels()
                    self.models.load_models(model_dir)
            except Exception as e:
                print(f"  [警告] 无法加载字段级预测器: {e}")
                self.models = None
    
    def enhance_batch(self, samples: List[Dict], verbose: bool = False) -> List[Dict]:
        """
        批量增强样本
        
        增强策略（优先级从高到低）：
        1. 整体判别器：过滤低真实度样本（如果启用）
        2. 字段级预测器：填充缺失字段、修正不一致（如果启用）
        """
        enhanced = samples
        
        # 步骤1：使用整体判别器过滤（推荐）
        if self.use_holistic and self.holistic_discriminator is not None:
            enhanced, scores = self.holistic_discriminator.filter_samples(
                enhanced, 
                threshold=self.holistic_threshold,
                verbose=verbose
            )
            if verbose:
                print(f"  [整体判别器] 平均真实度: {sum(scores)/len(scores):.3f}")
        
        # 步骤2：使用字段级预测器填充和修正（可选）
        if not self.use_discriminative or self.models is None:
            return enhanced
        
        enhanced_final = []
        filled_count = 0
        corrected_count = 0
        
        for sample in enhanced:  # 注意：处理已经过整体判别器过滤的样本
            enhanced_sample = sample.copy()
            
            # 1. 填充缺失的hours.per.week
            if not sample.get('hours.per.week') or sample.get('hours.per.week') == '':
                predicted_hours = self.models.predict(sample, 'hours')
                if predicted_hours:
                    enhanced_sample['hours.per.week'] = predicted_hours
                    filled_count += 1
            
            # 2. 验证并可能修正income（如果与其他特征不一致）
            if self._should_correct_income(sample):
                predicted_income = self.models.predict(sample, 'income')
                if predicted_income and predicted_income != sample.get('income'):
                    enhanced_sample['income'] = predicted_income
                    corrected_count += 1
            
            # 3. 填充缺失的occupation
            if not sample.get('occupation') or sample.get('occupation') == '?':
                predicted_occupation = self.models.predict(sample, 'occupation')
                if predicted_occupation:
                    enhanced_sample['occupation'] = predicted_occupation
                    filled_count += 1
            
            # 4. 填充缺失的workclass
            if not sample.get('workclass') or sample.get('workclass') == '?':
                predicted_workclass = self.models.predict(sample, 'workclass')
                if predicted_workclass:
                    enhanced_sample['workclass'] = predicted_workclass
                    filled_count += 1
            
            enhanced_final.append(enhanced_sample)
        
        if verbose and (filled_count > 0 or corrected_count > 0):
            print(f"  [字段级预测器] 填充缺失字段: {filled_count}, 修正不一致: {corrected_count}")
        
        return enhanced_final
    
    def _should_correct_income(self, sample: Dict) -> bool:
        """判断是否应该修正income字段"""
        # 检查明显的不一致
        age = sample.get('age', 0)
        education_num = sample.get('education.num', 0)
        hours = sample.get('hours.per.week', 0)
        income = sample.get('income', '')
        
        try:
            age = int(age)
            education_num = int(education_num)
            hours = int(hours)
        except:
            return False
        
        # 高学历 + 长工时 + 低收入 → 可能需要修正
        if education_num >= 14 and hours >= 40 and income == '<=50K':
            return True
        
        # 低学历 + 短工时 + 高收入 → 可能需要修正
        if education_num <= 9 and hours <= 30 and income == '>50K':
            return True
        
        return False
    
    def predict_missing_fields(self, sample: Dict) -> Dict:
        """预测并填充缺失字段"""
        if self.models is None:
            return sample
        
        enhanced = sample.copy()
        
        # 预测缺失字段
        if not sample.get('hours.per.week'):
            hours = self.models.predict(sample, 'hours')
            if hours:
                enhanced['hours.per.week'] = hours
        
        if not sample.get('occupation') or sample.get('occupation') == '?':
            occupation = self.models.predict(sample, 'occupation')
            if occupation:
                enhanced['occupation'] = occupation
        
        if not sample.get('workclass') or sample.get('workclass') == '?':
            workclass = self.models.predict(sample, 'workclass')
            if workclass:
                enhanced['workclass'] = workclass
        
        return enhanced
    
    def correct_inconsistencies(self, sample: Dict) -> Dict:
        """修正字段不一致"""
        if self.models is None:
            return sample
        
        enhanced = sample.copy()
        
        # 修正income（如果不一致）
        if self._should_correct_income(sample):
            predicted_income = self.models.predict(sample, 'income')
            if predicted_income:
                enhanced['income'] = predicted_income
        
        return enhanced


# ============================================================================
#                      Curation Pipeline
# ============================================================================

class AdultCurationPipeline:
    """
    策展流程管道
    
    完整流程：Filter → Reweight → Label Enhance → Auxiliary Enhance
    
    支持两种辅助模型模式：
    1. 整体判别器（推荐）- 自动学习分布，过滤低质量样本
    2. 字段级预测器 - 填充缺失字段，修正不一致
    """
    
    def __init__(self, use_reweighting: bool = True, use_enhancement: bool = True,
                 model_dir: str = None, use_holistic: bool = True, 
                 holistic_threshold: float = 0.5):
        """
        Args:
            use_reweighting: 是否使用重加权
            use_enhancement: 是否使用增强
            model_dir: 字段级预测器模型目录（可选）
            use_holistic: 是否使用整体判别器（推荐）
            holistic_threshold: 整体判别器的真实度阈值（0-1）
        """
        self.filter = AdultSampleFilter(strict=True)
        self.reweighter = AdultSampleReweighter()
        self.label_enhancer = AdultLabelEnhancer(smoothing=0.1)
        self.auxiliary_enhancer = AdultAuxiliaryModelEnhancer(
            model_dir=model_dir,
            use_discriminative=use_enhancement and model_dir is not None,
            use_holistic=use_holistic,
            holistic_threshold=holistic_threshold
        )
        
        self.use_reweighting = use_reweighting
        self.use_enhancement = use_enhancement
        self.use_holistic = use_holistic
    
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
