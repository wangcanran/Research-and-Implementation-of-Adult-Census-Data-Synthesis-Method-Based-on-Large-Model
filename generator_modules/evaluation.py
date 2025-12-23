"""
Evaluation Module - Benchmark Evaluation
从 data_generator.py 提取，修改为 Adult Census
"""

import numpy as np
from typing import List, Dict
from collections import Counter


class BenchmarkEvaluator:
    """
    Benchmark评估器：与真实数据对比（照搬data_generator.py核心）
    
    评估维度（修改为Adult Census）：
    1. 统计分布相似度（age, education, income, occupation等）
    2. 数值特征相似度（均值、方差、相关性）
    3. 业务逻辑一致性（教育-收入、工时-收入等）
    """
    
    def __init__(self, real_samples: List[Dict] = None):
        """
        初始化Benchmark评估器
        
        Args:
            real_samples: 真实数据样本（用作基准）
        """
        self.real_samples = real_samples or []
        self.real_stats = None
        
        if self.real_samples:
            self.real_stats = self._compute_statistics(self.real_samples)
    
    def load_real_samples(self, samples: List[Dict]) -> None:
        """加载真实样本数据"""
        self.real_samples = samples
        self.real_stats = self._compute_statistics(samples)
    
    def evaluate(self, generated_samples: List[Dict]) -> Dict:
        """
        综合评估：生成数据与真实数据的相似度（照搬data_generator.py）
        
        Returns:
            包含各维度相似度分数的字典
        """
        if not self.real_samples:
            return {
                "error": "未加载真实数据，无法进行Benchmark评估",
                "overall_similarity": 0
            }
        
        gen_stats = self._compute_statistics(generated_samples)
        
        results = {
            "distribution_similarity": self._compare_distributions(gen_stats, self.real_stats),
            "statistical_similarity": self._compare_statistics(gen_stats, self.real_stats),
            "logic_consistency": self._check_logic_consistency(generated_samples)
        }
        
        # 计算综合相似度（照搬data_generator.py的加权平均）
        results["overall_similarity"] = (
            results["distribution_similarity"]["score"] * 0.4 +
            results["statistical_similarity"]["score"] * 0.3 +
            results["logic_consistency"]["score"] * 0.3
        )
        
        return results
    
    def _compute_statistics(self, samples: List[Dict]) -> Dict:
        """计算样本的统计特征（修改为Adult Census）"""
        if not samples:
            return {}
        
        stats = {
            "count": len(samples),
            "distributions": {},
            "numerical": {}
        }
        
        # 1. 分布统计（修改为Adult Census字段）
        stats["distributions"]["income"] = self._get_distribution(samples, "income")
        stats["distributions"]["education"] = self._get_distribution(samples, "education")
        stats["distributions"]["occupation"] = self._get_distribution(samples, "occupation")
        stats["distributions"]["marital.status"] = self._get_distribution(samples, "marital.status")
        stats["distributions"]["sex"] = self._get_distribution(samples, "sex")
        stats["distributions"]["race"] = self._get_distribution(samples, "race")
        
        # 2. 数值特征统计
        ages = []
        for s in samples:
            if s.get("age"):
                try:
                    ages.append(int(s.get("age")))
                except (ValueError, TypeError):
                    pass
        
        hours = []
        for s in samples:
            if s.get("hours.per.week"):
                try:
                    hours.append(int(s.get("hours.per.week")))
                except (ValueError, TypeError):
                    pass
        
        edu_nums = []
        for s in samples:
            if s.get("education.num"):
                try:
                    edu_nums.append(int(s.get("education.num")))
                except (ValueError, TypeError):
                    pass
        
        if ages:
            stats["numerical"]["age"] = {
                "mean": np.mean(ages),
                "std": np.std(ages),
                "min": min(ages),
                "max": max(ages),
                "median": np.median(ages)
            }
        
        if hours:
            stats["numerical"]["hours.per.week"] = {
                "mean": np.mean(hours),
                "std": np.std(hours),
                "min": min(hours),
                "max": max(hours),
                "median": np.median(hours)
            }
        
        if edu_nums:
            stats["numerical"]["education.num"] = {
                "mean": np.mean(edu_nums),
                "std": np.std(edu_nums),
                "min": min(edu_nums),
                "max": max(edu_nums),
                "median": np.median(edu_nums)
            }
        
        return stats
    
    def _get_distribution(self, samples: List[Dict], field: str) -> Dict[str, float]:
        """获取字段的概率分布"""
        values = [s.get(field) for s in samples if s.get(field) and s.get(field) != '?']
        if not values:
            return {}
        
        counter = Counter(values)
        total = len(values)
        return {k: v / total for k, v in counter.items()}
    
    def _compare_distributions(self, gen_stats: Dict, real_stats: Dict) -> Dict:
        """比较分布相似度（照搬data_generator.py）"""
        results = {}
        score_sum = 0
        count = 0
        
        for field in ["income", "education", "occupation", "marital.status", "sex"]:
            gen_dist = gen_stats["distributions"].get(field, {})
            real_dist = real_stats["distributions"].get(field, {})
            
            if not gen_dist or not real_dist:
                continue
            
            # 计算JS散度（照搬data_generator.py）
            js_div = self._js_divergence(gen_dist, real_dist)
            similarity = 1 / (1 + js_div)  # 转换为相似度
            
            results[field] = {
                "js_divergence": js_div,
                "similarity": similarity,
                "generated_dist": gen_dist,
                "real_dist": real_dist
            }
            score_sum += similarity
            count += 1
        
        results["score"] = score_sum / count if count > 0 else 0
        return results
    
    def _compare_statistics(self, gen_stats: Dict, real_stats: Dict) -> Dict:
        """比较数值统计相似度（照搬data_generator.py）"""
        results = {}
        score_sum = 0
        count = 0
        
        for field in ["age", "hours.per.week", "education.num"]:
            gen_num = gen_stats["numerical"].get(field, {})
            real_num = real_stats["numerical"].get(field, {})
            
            if not gen_num or not real_num:
                continue
            
            # 比较均值和标准差
            mean_diff = abs(gen_num["mean"] - real_num["mean"]) / real_num["mean"]
            mean_sim = 1 / (1 + mean_diff)
            
            std_diff = abs(gen_num["std"] - real_num["std"]) / real_num["std"]
            std_sim = 1 / (1 + std_diff)
            
            results[field] = {
                "mean_similarity": mean_sim,
                "std_similarity": std_sim,
                "generated": gen_num,
                "real": real_num
            }
            score_sum += (mean_sim + std_sim) / 2
            count += 1
        
        results["score"] = score_sum / count if count > 0 else 0
        return results
    
    def _check_logic_consistency(self, samples: List[Dict]) -> Dict:
        """检查业务逻辑一致性（修改为Adult Census）"""
        total = len(samples)
        if total == 0:
            return {"score": 0}
        
        consistent_count = 0
        
        for sample in samples:
            is_consistent = True
            
            # 检查1: 年龄-婚姻
            try:
                age = int(sample.get("age", 0))
                marital = sample.get("marital.status", "")
                if age < 18 and marital in ["Married-civ-spouse"]:
                    is_consistent = False
            except:
                pass
            
            # 检查2: 婚姻-关系
            marital = sample.get("marital.status", "")
            relationship = sample.get("relationship", "")
            if marital == "Married-civ-spouse" and relationship not in ["Husband", "Wife"]:
                is_consistent = False
            
            # 检查3: 教育-收入大致合理（允许一定不合理）
            try:
                edu_num = int(sample.get("education.num", 9))
                income = sample.get("income", "<=50K")
                # 博士+低收入算不一致（但允许存在）
                # 这里只检查严重不合理的情况
                if edu_num >= 16 and income == "<=50K":
                    # 允许存在，不算完全不一致
                    pass
            except:
                pass
            
            if is_consistent:
                consistent_count += 1
        
        score = consistent_count / total
        
        return {
            "score": score,
            "consistent_count": consistent_count,
            "total": total,
            "consistency_rate": score
        }
    
    def _js_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """计算JS散度（照搬data_generator.py）"""
        # 合并所有key
        all_keys = set(p.keys()) | set(q.keys())
        
        # 填充缺失的key
        p_full = {k: p.get(k, 0.0) for k in all_keys}
        q_full = {k: q.get(k, 0.0) for k in all_keys}
        
        # 归一化
        p_sum = sum(p_full.values())
        q_sum = sum(q_full.values())
        
        if p_sum == 0 or q_sum == 0:
            return 1.0
        
        p_norm = {k: v / p_sum for k, v in p_full.items()}
        q_norm = {k: v / q_sum for k, v in q_full.items()}
        
        # 计算M = (P + Q) / 2
        m = {k: (p_norm[k] + q_norm[k]) / 2 for k in all_keys}
        
        # 计算KL(P||M)和KL(Q||M)
        kl_pm = sum(p_norm[k] * np.log(p_norm[k] / m[k]) 
                    if p_norm[k] > 0 and m[k] > 0 else 0 
                    for k in all_keys)
        
        kl_qm = sum(q_norm[k] * np.log(q_norm[k] / m[k]) 
                    if q_norm[k] > 0 and m[k] > 0 else 0 
                    for k in all_keys)
        
        # JS散度 = (KL(P||M) + KL(Q||M)) / 2
        return (kl_pm + kl_qm) / 2


class DirectEvaluator:
    """
    直接评估器（照搬data_generator.py）
    
    评估维度（完全照搬论文框架）：
    1. Faithfulness（忠实度）：约束检查 + Benchmark对比
    2. Diversity（多样性）：词汇多样性 + 样本独特性 + 分布覆盖度
    """
    
    def __init__(self):
        self.benchmark_evaluator = None  # 用于Faithfulness下的Benchmark评估
    
    def set_benchmark_evaluator(self, benchmark_evaluator):
        """设置Benchmark评估器（用于Faithfulness评估）"""
        self.benchmark_evaluator = benchmark_evaluator
    
    def evaluate(self, samples: List[Dict], target_dist: Dict = None) -> Dict:
        """
        综合评估（完全照搬data_generator.py框架）
        
        Returns:
            {
                "faithfulness": {...},  # 忠实度（约束检查 + Benchmark）
                "diversity": {...},     # 多样性
                "overall_score": float  # 综合分数
            }
        """
        if not samples:
            return {"overall_score": 0}
        
        results = {
            "faithfulness": self._evaluate_faithfulness(samples),
            "diversity": self._evaluate_diversity(samples, target_dist)
        }
        
        # 计算综合分数（忠实度60%，多样性40%）
        results["overall_score"] = (
            results["faithfulness"]["score"] * 0.6 +
            results["diversity"]["score"] * 0.4
        )
        
        return results
    
    def _evaluate_faithfulness(self, samples: List[Dict]) -> Dict:
        """
        忠实度评估（照搬data_generator.py）
        
        包括两部分：
        1. Constraint Check（约束检查）
        2. Benchmark Evaluation（与真实数据对比，如果有）
        """
        if not samples:
            return {"score": 0}
        
        # Part 1: 约束检查
        valid_count = 0
        issues_count = Counter()
        
        for sample in samples:
            is_valid = self._is_logic_consistent(sample)
            is_complete = self._is_format_correct(sample)
            
            if is_valid and is_complete:
                valid_count += 1
            else:
                if not is_complete:
                    issues_count["incomplete"] += 1
                if not is_valid:
                    issues_count["logic_error"] += 1
        
        constraint_score = valid_count / len(samples)
        
        # Part 2: Benchmark评估（如果有）
        benchmark_result = None
        benchmark_score = None
        
        if self.benchmark_evaluator and hasattr(self.benchmark_evaluator, 'real_samples') and self.benchmark_evaluator.real_samples:
            benchmark_result = self.benchmark_evaluator.evaluate(samples)
            benchmark_score = benchmark_result.get('overall_similarity', 0)
        
        # 计算综合忠实度分数
        if benchmark_score is not None:
            # 有Benchmark时：约束检查50% + Benchmark 50%
            overall_score = constraint_score * 0.5 + benchmark_score * 0.5
        else:
            # 无Benchmark时：只看约束检查
            overall_score = constraint_score
        
        result = {
            "constraint_check": {
                "score": constraint_score,
                "valid_count": valid_count,
                "total_count": len(samples),
                "common_issues": dict(issues_count.most_common(3))
            },
            "score": overall_score
        }
        
        # 如果有Benchmark评估，加入结果
        if benchmark_result:
            result["benchmark_evaluation"] = benchmark_result
        
        return result
    
    def _evaluate_diversity(self, samples: List[Dict], target_dist: Dict = None) -> Dict:
        """
        多样性评估（照搬data_generator.py）
        
        包括：
        1. 词汇多样性（Vocabulary Diversity）
        2. 样本独特性（Sample Uniqueness）
        3. 分布覆盖度（Distribution Coverage）
        """
        if not samples:
            return {"score": 0}
        
        # 1. 词汇多样性：统计不同值的数量
        unique_values = {
            "income": len(set(s.get("income") for s in samples if s.get("income"))),
            "education": len(set(s.get("education") for s in samples if s.get("education"))),
            "occupation": len(set(s.get("occupation") for s in samples if s.get("occupation"))),
            "age_range": len(set(self._get_age_range(s) for s in samples))
        }
        
        # 2. 样本独特性：检查重复样本（基于关键字段组合）
        sample_signatures = []
        for s in samples:
            sig = f"{s.get('age')}_{s.get('education')}_{s.get('occupation')}_{s.get('income')}"
            sample_signatures.append(sig)
        uniqueness = len(set(sample_signatures)) / len(sample_signatures) if sample_signatures else 0
        
        # 3. 分布覆盖度：检查是否覆盖目标分布
        coverage_score = 1.0
        if target_dist and "income" in target_dist:
            target_incomes = set(target_dist["income"].keys())
            actual_incomes = set(s.get("income") for s in samples if s.get("income"))
            coverage_score = len(actual_incomes & target_incomes) / len(target_incomes) if target_incomes else 1.0
        
        # 计算多样性分数（照搬data_generator.py权重）
        diversity_score = (
            min(unique_values["income"] / 2, 1.0) * 0.2 +        # 收入类别（2种）
            min(unique_values["education"] / 16, 1.0) * 0.25 +   # 教育水平（16种）
            min(unique_values["occupation"] / 14, 1.0) * 0.25 +  # 职业（14种）
            min(unique_values["age_range"] / 3, 1.0) * 0.1 +     # 年龄段（3种）
            uniqueness * 0.2                                      # 样本独特性
        )
        
        return {
            "score": diversity_score,
            "unique_values": unique_values,
            "uniqueness": uniqueness,
            "coverage": coverage_score
        }
    
    def _get_age_range(self, sample: Dict) -> str:
        """获取年龄段"""
        try:
            age = int(sample.get("age", 0))
            if age <= 30:
                return "young"
            elif age <= 55:
                return "middle"
            else:
                return "senior"
        except:
            return "middle"
    
    def _is_format_correct(self, sample: Dict) -> bool:
        """检查格式正确性"""
        required = ["age", "education", "income"]
        for field in required:
            if field not in sample or not sample[field] or sample[field] == '?':
                return False
        return True
    
    def _is_logic_consistent(self, sample: Dict) -> bool:
        """检查逻辑一致性"""
        try:
            age = int(sample.get("age", 0))
            marital = sample.get("marital.status", "")
            
            if age < 18 and marital in ["Married-civ-spouse"]:
                return False
            
            relationship = sample.get("relationship", "")
            if marital == "Married-civ-spouse" and relationship not in ["Husband", "Wife"]:
                return False
            
            return True
        except:
            return False
