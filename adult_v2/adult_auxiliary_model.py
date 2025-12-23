"""
Adult Census Data - Auxiliary Model (Discriminative Verifier)
完全照搬 data_generator.py 的 AuxiliaryModel 机制
"""

from typing import List, Dict
import random


class AdultAuxiliaryModel:
    """
    辅助模型（照搬 data_generator.py 的 AuxiliaryModel）
    
    核心功能：
    1. 判别式验证器：验证样本合理性
    2. 异常检测和修正
    3. 支持训练（从真实样本学习）
    """
    
    def __init__(self, use_discriminative: bool = True):
        """
        初始化辅助模型（照搬）
        
        Args:
            use_discriminative: 是否使用判别式模型（论文要求）
                              False则使用规则判断（向后兼容）
        """
        self.use_discriminative = use_discriminative
        self.discriminative_verifier = None
        self.trained = False
        
        if self.use_discriminative:
            print("[Auxiliary] Using discriminative model (as per paper)")
            # 这里可以加载真正的判别式模型
            # 暂时使用规则判断
            self.discriminative_verifier = AdultDiscriminativeVerifier()
        else:
            print("[Auxiliary] Using rule-based verification (fallback)")
    
    def train(self, real_samples: List[Dict]):
        """
        训练判别式辅助模型（照搬）
        
        Args:
            real_samples: 真实样本用于训练
        """
        if self.use_discriminative and self.discriminative_verifier:
            print(f"[Auxiliary] Training discriminative model, samples: {len(real_samples)}")
            self.discriminative_verifier.train(real_samples)
            self.trained = True
            print("[Auxiliary] Discriminative model training complete")
        else:
            # 规则判断无需训练
            self.trained = True
    
    def verify_with_classifier(self, samples: List[Dict]) -> List[Dict]:
        """
        使用判别式模型验证样本合理性（照搬）
        
        Args:
            samples: 待验证的样本
            
        Returns:
            验证后的样本（异常的会被修正或标记）
        """
        if self.use_discriminative and self.trained:
            # 使用判别式模型验证（论文要求）
            verified_samples = []
            for sample in samples:
                result = self.discriminative_verifier.verify(sample)
                
                # 应用修正建议（照搬）
                if result["overall"] == "anomalous" and result["corrections"]:
                    for correction in result["corrections"]:
                        # 解析修正建议
                        if " -> " in correction:
                            parts = correction.split(" -> ")
                            if len(parts) == 2:
                                field_part, value_part = parts
                                field = field_part.split(":")[0].strip()
                                new_value = value_part.split("(")[0].strip()
                                
                                # 应用修正
                                try:
                                    if field in ["age", "education.num", "hours.per.week"]:
                                        sample[field] = int(new_value)
                                    else:
                                        sample[field] = new_value
                                    sample["_discriminative_corrected"] = field
                                except:
                                    pass
                
                # 保存验证结果（照搬）
                sample["_auxiliary_result"] = result["overall"]
                sample["_auxiliary_confidence"] = result["confidence"]
                
                verified_samples.append(sample)
            
            return verified_samples
        else:
            # 降级：使用规则验证（照搬）
            verified_samples = []
            for sample in samples:
                verified = self._verify_logical_consistency(sample)
                verified_samples.append(verified)
            return verified_samples
    
    def _verify_logical_consistency(self, sample: Dict) -> Dict:
        """验证逻辑一致性（规则判断，照搬思路）"""
        try:
            # 年龄-婚姻一致性
            age = int(sample.get("age", 30))
            marital = sample.get("marital.status", "")
            
            # 18岁以下不应该已婚
            if age < 18 and marital in ["Married-civ-spouse", "Married-AF-spouse"]:
                sample["marital.status"] = "Never-married"
                sample["_auxiliary_fixed"] = "marital.status"
            
            # 婚姻-关系一致性
            relationship = sample.get("relationship", "")
            
            if marital == "Married-civ-spouse":
                if relationship not in ["Husband", "Wife"]:
                    sex = sample.get("sex", "Male")
                    sample["relationship"] = "Husband" if sex == "Male" else "Wife"
                    sample["_auxiliary_fixed"] = "relationship"
            
            if relationship in ["Husband", "Wife"]:
                if marital not in ["Married-civ-spouse", "Married-AF-spouse"]:
                    sample["marital.status"] = "Married-civ-spouse"
                    sample["_auxiliary_fixed"] = "marital.status"
            
            # 工时合理性
            hours = sample.get("hours.per.week")
            if hours:
                try:
                    hours_int = int(hours)
                    if hours_int < 1 or hours_int > 99:
                        sample["hours.per.week"] = 40  # 修正为标准工时
                        sample["_auxiliary_fixed"] = "hours.per.week"
                except:
                    sample["hours.per.week"] = 40
                    sample["_auxiliary_fixed"] = "hours.per.week"
            
            # 教育-收入合理性（软约束）
            edu_num = sample.get("education.num")
            income = sample.get("income")
            
            if edu_num and income:
                try:
                    edu_int = int(edu_num)
                    # 博士学位(16)+低收入，标记为可疑但不修正
                    if edu_int >= 16 and income == "<=50K":
                        sample["_auxiliary_warning"] = "high_education_low_income"
                except:
                    pass
        
        except:
            pass
        
        return sample


class AdultDiscriminativeVerifier:
    """
    判别式验证器（照搬 data_generator.py 的接口）
    
    功能：
    1. 训练：从真实样本学习分布
    2. 验证：检测生成样本是否异常
    3. 修正：提供修正建议
    """
    
    def __init__(self):
        self.trained = False
        self.real_stats = {}
    
    def train(self, real_samples: List[Dict]):
        """训练（从真实样本学习统计特征）"""
        # 学习关键统计特征
        import pandas as pd
        df = pd.DataFrame(real_samples)
        
        # 年龄分布
        self.real_stats['age_mean'] = df['age'].mean()
        self.real_stats['age_std'] = df['age'].std()
        
        # 工时分布
        if 'hours.per.week' in df.columns:
            self.real_stats['hours_mean'] = df['hours.per.week'].mean()
            self.real_stats['hours_std'] = df['hours.per.week'].std()
        
        # 收入分布
        if 'income' in df.columns:
            income_counts = df['income'].value_counts()
            total = len(df)
            self.real_stats['income_dist'] = {
                '<=50K': income_counts.get('<=50K', 0) / total,
                '>50K': income_counts.get('>50K', 0) / total
            }
        
        # 婚姻-年龄分布
        if 'marital.status' in df.columns:
            young = df[df['age'] <= 25]
            if len(young) > 0:
                married_ratio = len(young[young['marital.status'] == 'Married-civ-spouse']) / len(young)
                self.real_stats['young_married_ratio'] = married_ratio
        
        self.trained = True
    
    def verify(self, sample: Dict) -> Dict:
        """
        验证样本（照搬 data_generator.py 的返回格式）
        
        Returns:
            {
                "overall": "normal" or "anomalous",
                "confidence": float (0-1),
                "corrections": List[str]
            }
        """
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
            
            # 超过3个标准差
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
            
            # 年轻人高婚姻率异常
            if age <= 25 and marital == "Married-civ-spouse":
                young_married_ratio = self.real_stats.get('young_married_ratio', 0.1)
                if young_married_ratio < 0.15:  # 如果真实数据中年轻人结婚率很低
                    anomaly_score += 0.2
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
        
        # 检查收入-教育一致性
        try:
            edu_num = int(sample.get("education.num", 9))
            income = sample.get("income", "<=50K")
            
            # 高学历低收入
            if edu_num >= 16 and income == "<=50K":
                anomaly_score += 0.15  # 轻微异常
            
            # 低学历高收入
            if edu_num <= 8 and income == ">50K":
                anomaly_score += 0.15
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
