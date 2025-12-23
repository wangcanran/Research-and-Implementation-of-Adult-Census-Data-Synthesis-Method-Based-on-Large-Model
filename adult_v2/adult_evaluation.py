"""
Adult Census Data - Evaluation Stage
评估阶段 - 直接评估、基准对比、间接评估
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from .adult_task_spec import validate_sample


class AdultDirectEvaluator:
    """
    直接评估器
    
    评估维度：
    1. Format Correctness（格式正确性）
    2. Logical Consistency（逻辑一致性）
    3. Faithfulness（事实性）
    """
    
    def __init__(self):
        self.benchmark_evaluator = None
    
    def set_benchmark_evaluator(self, evaluator):
        """设置Benchmark评估器（用于Faithfulness评估）"""
        self.benchmark_evaluator = evaluator
    
    def evaluate(self, samples: List[Dict]) -> Dict:
        """
        评估生成的样本
        
        Returns:
            {
                "format_correctness": {...},
                "logical_consistency": {...},
                "faithfulness": {...},
                "overall_score": float
            }
        """
        result = {}
        
        # 1. Format Correctness
        format_result = self._evaluate_format(samples)
        result["format_correctness"] = format_result
        
        # 2. Logical Consistency
        consistency_result = self._evaluate_consistency(samples)
        result["logical_consistency"] = consistency_result
        
        # 3. Faithfulness (如果有benchmark)
        if self.benchmark_evaluator:
            faithfulness_result = self.benchmark_evaluator.evaluate_distribution_similarity(samples)
            result["faithfulness"] = faithfulness_result
        
        # 4. Overall Score
        result["overall_score"] = self._calculate_overall_score(result)
        
        return result
    
    def _evaluate_format(self, samples: List[Dict]) -> Dict:
        """评估格式正确性"""
        total = len(samples)
        valid_count = 0
        error_types = Counter()
        
        for sample in samples:
            is_valid, errors = validate_sample(sample)
            if is_valid:
                valid_count += 1
            else:
                for error in errors:
                    if "Missing field" in error:
                        error_types["missing_field"] += 1
                    elif "mapping error" in error:
                        error_types["mapping_error"] += 1
                    elif "out of range" in error:
                        error_types["range_error"] += 1
                    elif "inconsistency" in error:
                        error_types["inconsistency"] += 1
                    else:
                        error_types["other"] += 1
        
        return {
            "total_samples": total,
            "valid_samples": valid_count,
            "invalid_samples": total - valid_count,
            "validity_rate": valid_count / total if total > 0 else 0,
            "error_types": dict(error_types)
        }
    
    def _evaluate_consistency(self, samples: List[Dict]) -> Dict:
        """评估逻辑一致性"""
        total = len(samples)
        consistent_count = 0
        inconsistency_types = Counter()
        
        for sample in samples:
            is_consistent, issues = self._check_consistency(sample)
            if is_consistent:
                consistent_count += 1
            else:
                for issue in issues:
                    inconsistency_types[issue] += 1
        
        return {
            "total_samples": total,
            "consistent_samples": consistent_count,
            "consistency_rate": consistent_count / total if total > 0 else 0,
            "inconsistency_types": dict(inconsistency_types)
        }
    
    def _check_consistency(self, sample: Dict) -> Tuple[bool, List[str]]:
        """检查单个样本的逻辑一致性"""
        issues = []
        
        # 教育-收入一致性
        education = sample.get("education", "")
        income = sample.get("income", "")
        if education in ["Doctorate", "Prof-school", "Masters"]:
            if income == "<=50K":
                issues.append("high_education_low_income")
        
        # 年龄-婚姻一致性
        age = sample.get("age", 0)
        marital = sample.get("marital.status", "")
        try:
            age_val = int(age)
            if age_val < 20 and "Married" in marital:
                issues.append("too_young_married")
        except:
            pass
        
        # 工时-收入一致性
        hours = sample.get("hours.per.week", 0)
        try:
            hours_val = int(hours)
            if hours_val < 20 and income == ">50K":
                issues.append("low_hours_high_income")
        except:
            pass
        
        return len(issues) == 0, issues
    
    def _calculate_overall_score(self, result: Dict) -> float:
        """计算总体评分"""
        scores = []
        
        if "format_correctness" in result:
            scores.append(result["format_correctness"]["validity_rate"])
        
        if "logical_consistency" in result:
            scores.append(result["logical_consistency"]["consistency_rate"])
        
        if "faithfulness" in result and "overall_similarity" in result["faithfulness"]:
            scores.append(result["faithfulness"]["overall_similarity"])
        
        return np.mean(scores) if scores else 0.0


class AdultBenchmarkEvaluator:
    """
    基准对比评估器
    
    功能：
    1. 与真实数据分布对比
    2. 统计特征相似度
    3. 条件分布相似度
    """
    
    def __init__(self):
        self.real_samples = []
    
    def load_real_samples(self, samples: List[Dict]):
        """加载真实样本作为基准"""
        self.real_samples = samples
        print(f"  [BenchmarkEval] 加载 {len(samples)} 个真实样本作为基准")
    
    def evaluate_distribution_similarity(self, generated_samples: List[Dict]) -> Dict:
        """评估生成数据与真实数据的分布相似度"""
        if not self.real_samples:
            return {"error": "No real samples loaded"}
        
        result = {}
        
        # 1. 边缘分布相似度
        result["marginal_distributions"] = self._compare_marginal_distributions(generated_samples)
        
        # 2. 统计特征相似度
        result["statistical_features"] = self._compare_statistical_features(generated_samples)
        
        # 3. 关键组合分布
        result["combination_distributions"] = self._compare_combinations(generated_samples)
        
        # 4. 总体相似度
        result["overall_similarity"] = self._calculate_overall_similarity(result)
        
        return result
    
    def _compare_marginal_distributions(self, generated_samples: List[Dict]) -> Dict:
        """比较边缘分布"""
        comparisons = {}
        
        # 比较分类字段的分布
        categorical_fields = ["education", "occupation", "marital.status", "income", "sex", "race"]
        
        for field in categorical_fields:
            # 真实分布
            real_dist = self._get_distribution(self.real_samples, field)
            # 生成分布
            gen_dist = self._get_distribution(generated_samples, field)
            
            # 计算KL散度或JS散度
            similarity = self._calculate_distribution_similarity(real_dist, gen_dist)
            
            comparisons[field] = {
                "similarity": similarity,
                "real_top3": real_dist.most_common(3),
                "generated_top3": gen_dist.most_common(3)
            }
        
        return comparisons
    
    def _compare_statistical_features(self, generated_samples: List[Dict]) -> Dict:
        """比较统计特征"""
        comparisons = {}
        
        # 比较数值字段的统计量
        numerical_fields = ["age", "hours.per.week", "education.num"]
        
        for field in numerical_fields:
            real_values = [int(s.get(field, 0)) for s in self.real_samples if s.get(field)]
            gen_values = [int(s.get(field, 0)) for s in generated_samples if s.get(field)]
            
            if real_values and gen_values:
                comparisons[field] = {
                    "real_mean": np.mean(real_values),
                    "gen_mean": np.mean(gen_values),
                    "real_std": np.std(real_values),
                    "gen_std": np.std(gen_values),
                    "mean_diff": abs(np.mean(real_values) - np.mean(gen_values)),
                    "std_diff": abs(np.std(real_values) - np.std(gen_values))
                }
        
        return comparisons
    
    def _compare_combinations(self, generated_samples: List[Dict]) -> Dict:
        """比较关键组合的分布"""
        # 比较 (education, income) 组合
        real_combos = Counter()
        gen_combos = Counter()
        
        for sample in self.real_samples:
            edu = sample.get("education", "")
            income = sample.get("income", "")
            if edu and income:
                real_combos[f"{edu}_{income}"] += 1
        
        for sample in generated_samples:
            edu = sample.get("education", "")
            income = sample.get("income", "")
            if edu and income:
                gen_combos[f"{edu}_{income}"] += 1
        
        # 计算相似度
        similarity = self._calculate_distribution_similarity(real_combos, gen_combos)
        
        return {
            "education_income_similarity": similarity,
            "real_top5": real_combos.most_common(5),
            "generated_top5": gen_combos.most_common(5)
        }
    
    def _get_distribution(self, samples: List[Dict], field: str) -> Counter:
        """获取字段的分布"""
        dist = Counter()
        for sample in samples:
            value = sample.get(field)
            if value:
                dist[value] += 1
        return dist
    
    def _calculate_distribution_similarity(self, dist1: Counter, dist2: Counter) -> float:
        """计算两个分布的相似度（使用JS散度）"""
        # 获取所有类别
        all_categories = set(dist1.keys()) | set(dist2.keys())
        
        if not all_categories:
            return 1.0
        
        # 转换为概率分布
        total1 = sum(dist1.values())
        total2 = sum(dist2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        p = np.array([dist1.get(cat, 0) / total1 for cat in all_categories])
        q = np.array([dist2.get(cat, 0) / total2 for cat in all_categories])
        
        # 计算JS散度
        m = (p + q) / 2
        js_divergence = 0.5 * self._kl_divergence(p, m) + 0.5 * self._kl_divergence(q, m)
        
        # 转换为相似度 (0-1)
        similarity = 1.0 - min(js_divergence, 1.0)
        
        return similarity
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算KL散度"""
        # 避免log(0)
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return np.sum(p * np.log(p / q))
    
    def _calculate_overall_similarity(self, result: Dict) -> float:
        """计算总体相似度"""
        similarities = []
        
        # 边缘分布相似度
        if "marginal_distributions" in result:
            for field_result in result["marginal_distributions"].values():
                similarities.append(field_result["similarity"])
        
        # 组合分布相似度
        if "combination_distributions" in result:
            similarities.append(result["combination_distributions"]["education_income_similarity"])
        
        return np.mean(similarities) if similarities else 0.0


class AdultIndirectEvaluator:
    """
    间接评估器
    
    功能：
    1. 在下游任务上评估（如收入预测）
    2. 评估数据增强效果
    3. 评估模型泛化能力
    
    注：这是占位实现，实际需要训练下游任务模型
    """
    
    def __init__(self):
        self.downstream_models = {}
    
    def evaluate_on_downstream_task(self, train_samples: List[Dict], 
                                    test_samples: List[Dict],
                                    task: str = "income_prediction") -> Dict:
        """
        在下游任务上评估
        
        Args:
            train_samples: 用于训练的样本（可以是生成的）
            test_samples: 用于测试的样本（真实数据）
            task: 任务类型
        
        Returns:
            评估结果
        """
        # 占位实现：实际需要训练模型
        return {
            "task": task,
            "status": "not_implemented",
            "message": "需要训练下游任务模型"
        }
    
    def evaluate_augmentation_effect(self, original_samples: List[Dict],
                                     augmented_samples: List[Dict]) -> Dict:
        """评估数据增强效果"""
        # 占位实现
        return {
            "original_count": len(original_samples),
            "augmented_count": len(augmented_samples),
            "augmentation_ratio": len(augmented_samples) / len(original_samples) if original_samples else 0
        }
