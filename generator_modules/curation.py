"""
Curation Pipeline Module
从 data_generator.py 提取，修改为 Adult Census
包含完整的判别器机制
"""

from typing import List, Dict, Tuple
import random


class SampleFilter:
    """
    样本过滤器（照搬data_generator.py）
    
    功能：
    1. 启发式过滤
    2. 样本质量评分
    """
    
    def __init__(self, strict: bool = True):
        self.strict = strict
    
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """过滤样本（照搬data_generator.py）"""
        passed = []
        failed = []
        
        for sample in samples:
            is_valid, errors = self._validate_sample(sample)
            
            if is_valid or not self.strict:
                passed.append(sample)
            else:
                failed.append(sample)
        
        return passed, failed
    
    def _validate_sample(self, sample: Dict) -> Tuple[bool, List[str]]:
        """验证样本（修改为Adult Census）"""
        errors = []
        
        # 必需字段
        required_fields = ["age", "education", "income"]
        for field in required_fields:
            if field not in sample or sample[field] is None or sample[field] == "" or sample[field] == "?":
                errors.append(f"Missing {field}")
        
        # 年龄合理性
        if "age" in sample:
            try:
                age = int(sample["age"])
                if age < 17 or age > 90:
                    errors.append(f"Invalid age: {age}")
            except:
                errors.append(f"Invalid age type")
        
        # 收入值
        if "income" in sample:
            if sample["income"] not in ["<=50K", ">50K"]:
                errors.append(f"Invalid income")
        
        # 教育年数
        if "education.num" in sample:
            try:
                edu_num = int(sample["education.num"])
                if edu_num < 1 or edu_num > 16:
                    errors.append(f"Invalid education.num")
            except:
                errors.append(f"Invalid education.num type")
        
        # 逻辑一致性检查
        errors.extend(self._check_logical_consistency(sample))
        
        return len(errors) == 0, errors
    
    def _check_logical_consistency(self, sample: Dict) -> List[str]:
        """逻辑一致性检查（修改为Adult Census）"""
        errors = []
        
        # 年龄-婚姻
        try:
            age = int(sample.get("age", 0))
            marital = sample.get("marital.status", "")
            
            if age < 18 and marital in ["Married-civ-spouse", "Married-AF-spouse"]:
                errors.append("Age < 18 but married")
        except:
            pass
        
        # 婚姻-关系
        marital = sample.get("marital.status", "")
        relationship = sample.get("relationship", "")
        
        if marital == "Married-civ-spouse":
            if relationship not in ["Husband", "Wife"]:
                errors.append("Married but not Husband/Wife")
        
        if relationship in ["Husband", "Wife"]:
            if marital not in ["Married-civ-spouse", "Married-AF-spouse"]:
                errors.append("Husband/Wife but not married")
        
        # 工时合理性
        try:
            hours = int(sample.get("hours.per.week", 40))
            if hours < 1 or hours > 99:
                errors.append("Invalid hours range")
        except:
            pass
        
        return errors
    
    def evaluate_sample(self, sample: Dict) -> Tuple[float, Dict]:
        """评估样本质量（照搬data_generator.py）"""
        score = 1.0
        details = {}
        
        # 完整性（30%）
        required_fields = ["age", "workclass", "education", "marital.status",
                          "occupation", "relationship", "race", "sex",
                          "hours.per.week", "income"]
        present_fields = sum(1 for f in required_fields 
                           if f in sample and sample[f] and sample[f] != '?')
        completeness = present_fields / len(required_fields)
        details['completeness'] = completeness
        score *= (0.7 + 0.3 * completeness)
        
        # 逻辑一致性（50%）
        is_valid, errors = self._validate_sample(sample)
        if is_valid:
            consistency = 1.0
        else:
            consistency = max(0.0, 1.0 - len(errors) * 0.2)
        details['consistency'] = consistency
        score *= (0.5 + 0.5 * consistency)
        
        # 合理性（20%）
        reasonableness = self._check_reasonableness(sample)
        details['reasonableness'] = reasonableness
        score *= (0.8 + 0.2 * reasonableness)
        
        details['overall'] = score
        
        return score, details
    
    def _check_reasonableness(self, sample: Dict) -> float:
        """合理性检查（修改为Adult Census）"""
        score = 1.0
        
        # 教育-收入合理性
        try:
            edu_num = int(sample.get("education.num", 9))
            income = sample.get("income", "<=50K")
            
            # 高学历+低收入，不太合理
            if edu_num >= 14 and income == "<=50K":
                score *= 0.8
            
            # 低学历+高收入，可能但少见
            if edu_num <= 8 and income == ">50K":
                score *= 0.9
        except:
            pass
        
        # 工时-收入合理性
        try:
            hours = int(sample.get("hours.per.week", 40))
            income = sample.get("income", "<=50K")
            
            # 高工时+低收入
            if hours >= 50 and income == "<=50K":
                score *= 0.9
        except:
            pass
        
        return score


class AuxiliaryModel:
    """
    辅助模型（照搬data_generator.py的判别器机制）
    
    功能：
    1. 验证样本合理性
    2. 异常检测和修正
    """
    
    def __init__(self):
        self.trained = False
        self.real_stats = {}
    
    def train(self, real_samples: List[Dict]):
        """训练（从真实样本学习）"""
        import pandas as pd
        df = pd.DataFrame(real_samples)
        
        # 学习关键统计特征
        self.real_stats['age_mean'] = df['age'].mean()
        self.real_stats['age_std'] = df['age'].std()
        
        if 'hours.per.week' in df.columns:
            self.real_stats['hours_mean'] = df['hours.per.week'].mean()
            self.real_stats['hours_std'] = df['hours.per.week'].std()
        
        if 'income' in df.columns:
            income_counts = df['income'].value_counts()
            total = len(df)
            self.real_stats['income_dist'] = {
                '<=50K': income_counts.get('<=50K', 0) / total,
                '>50K': income_counts.get('>50K', 0) / total
            }
        
        self.trained = True
    
    def verify_with_classifier(self, samples: List[Dict]) -> List[Dict]:
        """使用判别器验证样本（照搬data_generator.py）"""
        verified_samples = []
        
        for sample in samples:
            result = self._verify_sample(sample)
            
            # 应用修正建议
            if result["overall"] == "anomalous" and result["corrections"]:
                for correction in result["corrections"]:
                    if " -> " in correction:
                        parts = correction.split(" -> ")
                        if len(parts) == 2:
                            field_part, value_part = parts
                            field = field_part.split(":")[0].strip()
                            new_value = value_part.split("(")[0].strip()
                            
                            try:
                                if field in ["age", "education.num", "hours.per.week"]:
                                    sample[field] = int(new_value)
                                else:
                                    sample[field] = new_value
                                sample["_discriminative_corrected"] = field
                            except:
                                pass
            
            # 保存验证结果
            sample["_auxiliary_result"] = result["overall"]
            sample["_auxiliary_confidence"] = result["confidence"]
            
            verified_samples.append(sample)
        
        return verified_samples
    
    def _verify_sample(self, sample: Dict) -> Dict:
        """验证单个样本（照搬data_generator.py）"""
        if not self.trained:
            return {
                "overall": "normal",
                "confidence": 1.0,
                "corrections": []
            }
        
        anomaly_score = 0.0
        corrections = []
        
        # 检查年龄
        try:
            age = int(sample.get("age", 0))
            age_mean = self.real_stats.get('age_mean', 40)
            age_std = self.real_stats.get('age_std', 15)
            
            if abs(age - age_mean) > 3 * age_std:
                anomaly_score += 0.3
                suggested_age = int(age_mean + random.uniform(-age_std, age_std))
                corrections.append(f"age: {age} -> {suggested_age} (outlier)")
        except:
            pass
        
        # 检查年龄-婚姻
        try:
            age = int(sample.get("age", 30))
            marital = sample.get("marital.status", "")
            
            if age < 18 and marital in ["Married-civ-spouse"]:
                anomaly_score += 0.4
                corrections.append(f"marital.status: {marital} -> Never-married (age < 18)")
        except:
            pass
        
        # 检查工时
        try:
            hours = int(sample.get("hours.per.week", 40))
            if hours < 1 or hours > 99:
                anomaly_score += 0.3
                corrections.append(f"hours.per.week: {hours} -> 40 (out of range)")
        except:
            pass
        
        # 判定
        if anomaly_score >= 0.5:
            overall = "anomalous"
            confidence = min(1.0, anomaly_score)
        else:
            overall = "normal"
            confidence = 1.0 - anomaly_score
        
        return {
            "overall": overall,
            "confidence": confidence,
            "corrections": corrections
        }


class CurationPipeline:
    """
    策展管道（照搬data_generator.py）
    
    功能：
    1. 过滤
    2. 辅助模型验证（判别器机制）
    """
    
    def __init__(self, use_filter: bool = True, use_auxiliary: bool = True):
        self.filter = SampleFilter(strict=True)
        self.use_filter = use_filter
        self.use_auxiliary = use_auxiliary
        
        if use_auxiliary:
            self.auxiliary_model = AuxiliaryModel()
        else:
            self.auxiliary_model = None
    
    def train_auxiliary(self, real_samples: List[Dict]):
        """训练辅助模型（照搬data_generator.py）"""
        if self.auxiliary_model:
            self.auxiliary_model.train(real_samples)
    
    def curate(self, samples: List[Dict]) -> Dict:
        """
        策展（照搬data_generator.py的完整流程）
        
        Returns:
            {
                "curated_samples": List[Dict],
                "filter_stats": Dict,
                "auxiliary_stats": Dict
            }
        """
        curated_samples = samples.copy()
        stats = {}
        
        # Step 1: 过滤
        if self.use_filter:
            passed, failed = self.filter.filter_samples(curated_samples)
            curated_samples = passed
            
            stats['filter_stats'] = {
                'original': len(samples),
                'passed': len(passed),
                'failed': len(failed),
                'pass_rate': len(passed) / len(samples) if samples else 0
            }
        
        # Step 2: 辅助模型验证（判别器机制）
        if self.use_auxiliary and self.auxiliary_model and self.auxiliary_model.trained:
            curated_samples = self.auxiliary_model.verify_with_classifier(curated_samples)
            
            corrected_count = sum(1 for s in curated_samples if '_discriminative_corrected' in s)
            anomalous_count = sum(1 for s in curated_samples if s.get('_auxiliary_result') == 'anomalous')
            
            stats['auxiliary_stats'] = {
                'total': len(curated_samples),
                'corrected': corrected_count,
                'anomalous': anomalous_count
            }
        
        return {
            "curated_samples": curated_samples,
            "filter_stats": stats.get('filter_stats', {}),
            "auxiliary_stats": stats.get('auxiliary_stats', {})
        }
