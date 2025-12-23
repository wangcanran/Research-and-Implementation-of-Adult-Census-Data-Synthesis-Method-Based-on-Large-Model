import json
import random
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from openai import OpenAI
import numpy as np
import pandas as pd

try:
    import adult_config as config
except ImportError:
    print("[Warning] adult_config.py not found, using defaults")
    class config:
        OPENAI_API_KEY = "your-api-key-here"
        OPENAI_API_BASE = "https://api.openai.com/v1"
        REQUEST_TIMEOUT = 60
        FIXED_MODEL_NAME = "gpt-4o-mini"
        DEFAULT_TEMPERATURE = 0.7
        DEFAULT_MAX_TOKENS = 800


# ============================================================================
#                           基础配置与客户端
# ============================================================================

client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_API_BASE,
    timeout=config.REQUEST_TIMEOUT
)

MODEL_NAME = config.FIXED_MODEL_NAME


# ============================================================================
#                      I. GENERATION - 生成阶段
# ============================================================================

# ---------------------- 1.1 Task Specification ----------------------

ADULT_TASK_SPECIFICATION = """
你是一个人口普查数据生成专家。你的任务是生成符合真实人口统计学逻辑的成人收入数据。

## Adult Census Income 数据表结构

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| age | Integer | 年龄 | 17-90岁，符合人口分布 |
| workclass | String | 工作类型 | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked, ? |
| fnlwgt | Integer | 最终权重 | 人口统计权重，范围12285-1484705 |
| education | String | 教育程度 | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool |
| education.num | Integer | 教育年限 | 1-16年，与education对应 |
| marital.status | String | 婚姻状况 | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse |
| occupation | String | 职业 | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces, ? |
| relationship | String | 家庭关系 | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried |
| race | String | 种族 | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black |
| sex | String | 性别 | Female, Male |
| capital.gain | Integer | 资本收益 | 0-99999，大多数为0 |
| capital.loss | Integer | 资本损失 | 0-4356，大多数为0 |
| hours.per.week | Integer | 每周工作时长 | 1-99小时，大多数为40 |
| native.country | String | 原籍国家 | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands, ? |
| income | String | 收入类别 | <=50K, >50K |

## 核心业务规则

1. **年龄-教育-职业关联**:
   - 年轻人(17-25): 多为在校或初级职业，Never-married居多
   - 中年人(26-55): 多为成熟职业，Married居多
   - 老年人(56+): 可能退休或高级职业，Married/Widowed

2. **教育-收入-职业关联**:
   - Doctorate/Prof-school/Masters → 高收入(>50K)概率高，职业多为Prof-specialty/Exec-managerial
   - HS-grad/Some-college → 中等收入，职业多样化
   - <HS-grad → 低收入(<=50K)概率高，职业多为Other-service/Craft-repair

3. **性别-婚姻-关系关联**:
   - Male + Married-civ-spouse → Husband
   - Female + Married-civ-spouse → Wife
   - Never-married → Not-in-family/Own-child
   - Divorced/Separated/Widowed → Not-in-family/Unmarried

4. **工作时长-收入关联**:
   - hours.per.week >= 40 且高学历 → >50K概率高
   - hours.per.week < 35 → <=50K概率高

5. **资本收益/损失**:
   - 大多数人为0（约90%）
   - 高收入人群有capital.gain/loss的概率更高

6. **教育编码对应**:
   - Preschool: 1, 1st-4th: 2, 5th-6th: 3, 7th-8th: 4, 9th: 5, 10th: 6, 11th: 7, 12th: 8
   - HS-grad: 9, Some-college: 10, Assoc-voc: 11, Assoc-acdm: 12
   - Bachelors: 13, Masters: 14, Prof-school: 15, Doctorate: 16
"""

# 教育等级映射
EDUCATION_MAPPING = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8,
    "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
}

# 反向映射
EDUCATION_NUM_TO_NAME = {v: k for k, v in EDUCATION_MAPPING.items()}


# ---------------------- 1.2 Generation Conditions ----------------------

@dataclass
class GenerationCondition:
    """生成条件 - Conditioning Scope & Values"""
    
    # 条件范围 (Conditioning Scope)
    age_range: Optional[str] = None          # 年龄范围：young, middle, senior
    education_level: Optional[str] = None    # 教育水平：low, medium, high
    income_class: Optional[str] = None       # 收入类别：<=50K, >50K
    gender: Optional[str] = None             # 性别：Male, Female
    marital_status: Optional[str] = None     # 婚姻状况
    
    # 条件值 (Conditioning Values)
    target_distribution: Optional[Dict] = None  # 目标分布
    
    def to_prompt(self) -> str:
        """将条件转换为提示词"""
        parts = []
        
        if self.age_range:
            parts.append(f"年龄范围: {self.age_range}")
        if self.education_level:
            parts.append(f"教育水平: {self.education_level}")
        if self.income_class:
            parts.append(f"收入类别: {self.income_class}")
        if self.gender:
            parts.append(f"性别: {self.gender}")
        if self.marital_status:
            parts.append(f"婚姻状况: {self.marital_status}")
        
        return "；".join(parts) if parts else "无特殊约束"


# ---------------------- 1.3 In-Context Demonstrations ----------------------

class DemonstrationManager:
    """上下文示例管理器"""
    
    # 预定义高质量示例
    DEMONSTRATIONS = {
        "high_income_educated": {
            "age": 45,
            "workclass": "Private",
            "fnlwgt": 234567,
            "education": "Masters",
            "education.num": 14,
            "marital.status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital.gain": 15024,
            "capital.loss": 0,
            "hours.per.week": 50,
            "native.country": "United-States",
            "income": ">50K"
        },
        "low_income_young": {
            "age": 22,
            "workclass": "Private",
            "fnlwgt": 189450,
            "education": "Some-college",
            "education.num": 10,
            "marital.status": "Never-married",
            "occupation": "Sales",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital.gain": 0,
            "capital.loss": 0,
            "hours.per.week": 35,
            "native.country": "United-States",
            "income": "<=50K"
        },
        "middle_income": {
            "age": 38,
            "workclass": "Private",
            "fnlwgt": 215646,
            "education": "HS-grad",
            "education.num": 9,
            "marital.status": "Married-civ-spouse",
            "occupation": "Craft-repair",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital.gain": 0,
            "capital.loss": 0,
            "hours.per.week": 40,
            "native.country": "United-States",
            "income": "<=50K"
        }
    }
    
    def __init__(self, use_heuristic: bool = True):
        self.demonstrations = self.DEMONSTRATIONS
        self.real_samples = []
        self.use_heuristic = use_heuristic
        self.sample_quality_cache = {}

    def load_samples_from_file(self, file_path: str):
        """从CSV文件加载真实样本作为示例"""
        try:
            df = pd.read_csv(file_path)
            self.real_samples = df.to_dict('records')
            print(f"  [提示] 已加载 {len(self.real_samples)} 条真实样本作为示例池")
        except Exception as e:
            print(f"  [警告] 加载样本文件失败: {e}")

    def select_demonstrations(self, condition: GenerationCondition, k: int = 2) -> List[Dict]:
        """示例选择策略"""
        if self.use_heuristic and self.real_samples:
            return self._heuristic_select(condition, k)
        
        # 简单随机选择
        if self.real_samples:
            try:
                return random.sample(self.real_samples, min(k, len(self.real_samples)))
            except:
                pass

        selected = []
        
        # 根据条件选择最相关的示例
        if condition.income_class == ">50K":
            selected.append(self.demonstrations["high_income_educated"])
        elif condition.income_class == "<=50K":
            if condition.age_range == "young":
                selected.append(self.demonstrations["low_income_young"])
            else:
                selected.append(self.demonstrations["middle_income"])
        else:
            # 默认混合示例
            selected.extend([
                self.demonstrations["high_income_educated"],
                self.demonstrations["middle_income"]
            ])
        
        return selected[:k]
    
    def _heuristic_select(self, condition: GenerationCondition, k: int) -> List[Dict]:
        """启发式高质量样本选择"""
        qualified_samples = []
        for sample in self.real_samples:
            quality_score = self._calculate_quality_score(sample)
            
            if quality_score < 0.5:
                continue
                
            qualified_samples.append((sample, quality_score))
        
        if not qualified_samples:
            return random.sample(self.real_samples, min(k, len(self.real_samples)))
        
        # 计算相似度并排序
        scored_samples = []
        for sample, quality in qualified_samples:
            similarity = self._calculate_similarity(sample, condition)
            uncertainty = self._calculate_uncertainty(sample)
            
            final_score = quality * 0.4 + similarity * 0.4 + uncertainty * 0.2
            scored_samples.append((sample, final_score, uncertainty))
        
        # 不确定性过滤
        filtered_samples = [(s, score) for s, score, unc in scored_samples if unc > 0.2]
        
        if not filtered_samples:
            filtered_samples = [(s, score) for s, score, _ in scored_samples]
        
        # 按得分排序
        filtered_samples.sort(key=lambda x: x[1], reverse=True)
        
        # 多样性采样
        candidate_pool = filtered_samples[:min(k * 2, len(filtered_samples))]
        selected = self._diverse_sampling([s for s, _ in candidate_pool], k)
        
        return selected
    
    def _calculate_quality_score(self, sample: Dict) -> float:
        """样本质量评分"""
        score = 1.0
        
        # 字段完整性检查
        required_fields = ["age", "education", "occupation", "hours.per.week", "income"]
        missing = sum(1 for f in required_fields if not sample.get(f) or sample.get(f) == "?")
        completeness = 1.0 - (missing / len(required_fields))
        score *= (0.3 * completeness + 0.7)
        
        # 逻辑一致性检查
        try:
            age = int(sample.get("age", 0))
            if 17 <= age <= 90:
                score *= 1.0
            else:
                score *= 0.3
        except:
            score *= 0.5
        
        # 教育-收入一致性
        try:
            education = sample.get("education", "")
            income = sample.get("income", "")
            
            high_edu = education in ["Doctorate", "Prof-school", "Masters", "Bachelors"]
            high_income = income == ">50K"
            
            # 高教育与高收入正相关
            if high_edu == high_income:
                score *= 1.0
            else:
                score *= 0.8
        except:
            score *= 0.7
        
        # 婚姻-关系一致性
        try:
            marital = sample.get("marital.status", "")
            relationship = sample.get("relationship", "")
            sex = sample.get("sex", "")
            
            if marital == "Married-civ-spouse":
                if (sex == "Male" and relationship == "Husband") or \
                   (sex == "Female" and relationship == "Wife"):
                    score *= 1.0
                else:
                    score *= 0.6
        except:
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def _calculate_similarity(self, sample: Dict, condition: GenerationCondition) -> float:
        """计算样本与生成条件的相似度"""
        similarity = 0.0
        
        # 年龄范围匹配
        if condition.age_range:
            try:
                age = int(sample.get("age", 0))
                age_match = {
                    "young": 17 <= age <= 30,
                    "middle": 31 <= age <= 55,
                    "senior": 56 <= age <= 90
                }
                if age_match.get(condition.age_range, False):
                    similarity += 0.25
            except:
                pass
        else:
            similarity += 0.125
        
        # 教育水平匹配
        if condition.education_level:
            try:
                edu_num = int(sample.get("education.num", 0))
                edu_match = {
                    "low": edu_num <= 8,
                    "medium": 9 <= edu_num <= 12,
                    "high": edu_num >= 13
                }
                if edu_match.get(condition.education_level, False):
                    similarity += 0.25
            except:
                pass
        else:
            similarity += 0.125
        
        # 收入类别匹配
        if condition.income_class:
            if sample.get("income") == condition.income_class:
                similarity += 0.3
        else:
            similarity += 0.15
        
        # 性别匹配
        if condition.gender:
            if sample.get("sex") == condition.gender:
                similarity += 0.2
        else:
            similarity += 0.1
        
        return max(0.0, min(1.0, similarity))
    
    def _calculate_uncertainty(self, sample: Dict) -> float:
        """计算样本不确定性（难度）"""
        uncertainty = 0.5
        
        # 稀有组合增加难度
        try:
            age = int(sample.get("age", 0))
            education = sample.get("education", "")
            
            # 年轻但高学历（稀有）
            if age < 25 and education in ["Doctorate", "Prof-school"]:
                uncertainty += 0.2
            
            # 老年但低学历（常见，降低难度）
            if age > 60 and education in ["HS-grad", "Some-college"]:
                uncertainty -= 0.1
        except:
            pass
        
        # 资本收益/损失存在（稀有）
        try:
            if int(sample.get("capital.gain", 0)) > 0:
                uncertainty += 0.15
            if int(sample.get("capital.loss", 0)) > 0:
                uncertainty += 0.15
        except:
            pass
        
        # 非美国籍（稀有）
        if sample.get("native.country") not in ["United-States", "?"]:
            uncertainty += 0.1
        
        return max(0.0, min(1.0, uncertainty))
    
    def _diverse_sampling(self, candidates: List[Dict], k: int) -> List[Dict]:
        """多样性采样"""
        if len(candidates) <= k:
            return candidates
        
        selected = [candidates[0]]
        candidates = candidates[1:]
        
        while len(selected) < k and candidates:
            max_min_distance = -1
            best_candidate = None
            best_idx = -1
            
            for idx, candidate in enumerate(candidates):
                min_distance = min(
                    self._sample_distance(candidate, s) for s in selected
                )
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate:
                selected.append(best_candidate)
                candidates.pop(best_idx)
            else:
                break
        
        return selected
    
    def _sample_distance(self, s1: Dict, s2: Dict) -> float:
        """计算两个样本之间的距离"""
        distance = 0.0
        
        # 年龄差异
        try:
            a1 = int(s1.get("age", 0))
            a2 = int(s2.get("age", 0))
            distance += abs(a1 - a2) / 90.0
        except:
            pass
        
        # 教育差异
        try:
            e1 = int(s1.get("education.num", 0))
            e2 = int(s2.get("education.num", 0))
            distance += abs(e1 - e2) / 16.0
        except:
            pass
        
        # 收入类别差异
        if s1.get("income") != s2.get("income"):
            distance += 0.5
        
        return distance / 3.0
    
    def format_demonstrations(self, demos: List[Dict]) -> str:
        """格式化示例为提示词"""
        formatted = "## 参考示例\n\n"
        for i, demo in enumerate(demos, 1):
            formatted += f"### 示例 {i}\n```json\n{json.dumps(demo, ensure_ascii=False, indent=2)}\n```\n\n"
        return formatted


# ---------------------- 1.4 Sample-Wise Decomposition ----------------------

class SampleWiseDecomposer:
    """样本级别分解 - 将样本拆分为多个字段组分步生成"""
    
    # 字段分组
    FIELD_GROUPS = {
        "demographics": ["age", "sex", "race", "native.country"],
        "education": ["education", "education.num"],
        "work": ["workclass", "occupation", "hours.per.week"],
        "family": ["marital.status", "relationship"],
        "financial": ["capital.gain", "capital.loss", "fnlwgt"],
        "outcome": ["income"]
    }
    
    def __init__(self, learned_stats: Dict = None):
        self.learned_stats = learned_stats if learned_stats is not None else {}
        self.generation_cache = {
            'ages': [],
            'hours': [],
            'count': 0
        }
    
    def update_cache(self, sample: Dict):
        """【记忆机制】更新生成缓存（每生成一个样本后调用）"""
        try:
            if 'age' in sample:
                self.generation_cache['ages'].append(int(sample['age']))
            if 'hours.per.week' in sample:
                self.generation_cache['hours'].append(int(sample['hours.per.week']))
            if 'education.num' in sample:
                self.generation_cache.setdefault('educations', []).append(int(sample['education.num']))
            if 'income' in sample:
                self.generation_cache.setdefault('incomes', []).append(sample['income'])
            self.generation_cache['count'] += 1
        except:
            pass
    
    def reset_cache(self):
        """【记忆机制】重置缓存（开始新一批生成时调用）"""
        self.generation_cache = {
            'ages': [],
            'hours': [],
            'educations': [],
            'incomes': [],
            'count': 0
        }
    
    def get_cache_stats(self) -> Dict:
        """【记忆机制】获取当前缓存的统计信息"""
        ages = self.generation_cache.get('ages', [])
        hours = self.generation_cache.get('hours', [])
        educations = self.generation_cache.get('educations', [])
        incomes = self.generation_cache.get('incomes', [])
        
        if not ages:
            return {}
        
        stats = {
            'age_mean': int(np.mean(ages)),
            'age_std': int(np.std(ages)),
            'count': self.generation_cache['count']
        }
        
        if hours:
            stats['hours_mean'] = int(np.mean(hours))
            stats['hours_std'] = int(np.std(hours))
        
        if educations:
            stats['education_mean'] = int(np.mean(educations))
        
        if incomes:
            stats['high_income_ratio'] = incomes.count('>50K') / len(incomes)
        
        return stats
    
    def decompose_and_generate(self, condition: GenerationCondition) -> Dict:
        """分步生成完整样本"""
        sample = {}
        context = {"condition": condition}
        
        # 按依赖顺序生成
        generation_order = ["demographics", "education", "work", "family", "financial", "outcome"]
        
        for group_name in generation_order:
            group_data = self._generate_field_group(group_name, sample, context)
            sample.update(group_data)
        
        return sample
    
    def _generate_field_group(self, group_name: str, current_sample: Dict, context: Dict) -> Dict:
        """生成单个字段组"""
        condition = context["condition"]
        
        prompts = {
            "demographics": self._get_demographics_prompt(condition),
            "education": self._get_education_prompt(condition, current_sample),
            "work": self._get_work_prompt(condition, current_sample),
            "family": self._get_family_prompt(current_sample),
            "financial": self._get_financial_prompt(current_sample),
            "outcome": self._get_outcome_prompt(condition, current_sample)
        }
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是人口统计数据字段生成器。只输出JSON对象，不要其他文字。"},
                {"role": "user", "content": prompts[group_name]}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        content = self._clean_json(content)
        
        try:
            result = json.loads(content)
            
            # 后处理：确保education和education.num一致
            if group_name == "education" and "education" in result:
                edu_name = result["education"]
                if edu_name in EDUCATION_MAPPING:
                    result["education.num"] = EDUCATION_MAPPING[edu_name]
            
            # 【新增】条件依赖验证：检查LLM生成是否符合条件分布
            if not self._verify_conditional_dependency(group_name, result, current_sample, condition):
                # 如果不符合，使用fallback的条件分布采样
                return self._fallback_generate(group_name, condition, current_sample)
            
            return result
        except json.JSONDecodeError:
            return self._fallback_generate(group_name, condition, current_sample)
    
    def _verify_conditional_dependency(self, group_name: str, result: Dict, 
                                       current_sample: Dict, condition: GenerationCondition) -> bool:
        """验证LLM生成的字段是否符合条件依赖关系
        
        Returns:
            True: 符合条件依赖，可以使用
            False: 不符合，需要回退到fallback
        """
        # 如果没有学习统计信息，跳过验证
        if not self.learned_stats:
            return True
        
        try:
            # 1. 验证education与age的依赖 P(education|age)
            if group_name == "education":
                age = current_sample.get("age", 30)
                education = result.get("education")
                education_num = result.get("education.num", 0)
                
                # 检查年龄-教育合理性：年轻人不应该有博士学位
                if age < 25 and education in ["Doctorate", "Prof-school"]:
                    return False  # 不合理
                if age > 70 and education_num <= 6:  # 老年人不太可能只朆10年级以下
                    pass  # 这个可能存在，不强制拒绝
                
                return True
            
            # 2. 验证occupation与education的依赖 P(occupation|education)
            elif group_name == "work":
                education = current_sample.get("education", "")
                occupation = result.get("occupation", "")
                hours = result.get("hours.per.week", 40)
                
                # 高学历不应该从事低端职业
                if education in ["Doctorate", "Prof-school", "Masters"]:
                    low_end_occupations = ["Handlers-cleaners", "Priv-house-serv", "Farming-fishing"]
                    if occupation in low_end_occupations:
                        return False
                
                # 工作时长合理性
                if not (1 <= hours <= 99):
                    return False
                
                return True
            
            # 3. 验证relationship与marital+sex的依赖
            elif group_name == "family":
                sex = current_sample.get("sex", "Male")
                marital = result.get("marital.status", "")
                relationship = result.get("relationship", "")
                
                # 强制一致性：已婚男性必须为Husband
                if marital == "Married-civ-spouse":
                    if sex == "Male" and relationship != "Husband":
                        return False
                    if sex == "Female" and relationship != "Wife":
                        return False
                
                return True
            
            # 4. 验证income与education/hours的依赖
            elif group_name == "outcome":
                education = current_sample.get("education", "")
                hours = current_sample.get("hours.per.week", 40)
                income = result.get("income", "")
                
                # 使用学到的条件分布验证合理性
                if 'income_given_education' in self.learned_stats:
                    if education in self.learned_stats['income_given_education']:
                        prob_high = self.learned_stats['income_given_education'][education]['>50K']
                        
                        # 如果教育程度很高（>70%高收入），但生成了低收入，且工时正常，则不合理
                        if prob_high > 0.7 and income == "<=50K" and hours >= 40:
                            if random.random() < 0.7:  # 70%概率拒绝
                                return False
                
                return True
            
            # 其他字段组默认通过
            return True
            
        except Exception:
            # 验证失败时，保守地通过
            return True
    
    def _clean_json(self, content: str) -> str:
        """清理JSON格式"""
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        return content.strip()
    
    def _get_demographics_prompt(self, condition: GenerationCondition) -> str:
        # 【引导式生成】从条件分布采样得到建议值，放入Prompt中引导LLM
        if self.learned_stats and 'age' in self.learned_stats:
            age_mean = self.learned_stats['age']['mean']
            age_std = self.learned_stats['age']['std']
            age_min = self.learned_stats['age']['min']
            age_max = self.learned_stats['age']['max']
            
            if condition.age_range == "young":
                age_mean = 24
            elif condition.age_range == "middle":
                age_mean = 42
            elif condition.age_range == "senior":
                age_mean = 65
            
            suggested_age = int(np.clip(np.random.normal(age_mean, age_std), age_min, age_max))
        else:
            age_ranges = {"young": (17, 30), "middle": (31, 55), "senior": (56, 90)}
            age_range = age_ranges.get(condition.age_range, (17, 90))
            suggested_age = random.randint(*age_range)
        
        gender_hint = f"性别为{condition.gender}" if condition.gender else "性别随机（Male/Female）"
        
        return f"""生成人口统计学字段:
- age: 年龄，建议值 {suggested_age} 岁（可微调±2岁）
- sex: {gender_hint}
- race: 种族，White占85%，其他为Asian-Pac-Islander/Black/Other/Amer-Indian-Eskimo
- native.country: 原籍国家，United-States占90%，其他为Mexico/Philippines/Germany等

输出JSON对象:"""

    def _get_education_prompt(self, condition: GenerationCondition, sample: Dict) -> str:
        age = sample.get("age", 30)
        age_range = "young" if age <= 30 else ("middle" if age <= 55 else "senior")
        
        # 【引导式生成】从P(education|age)采样建议值
        if self.learned_stats and 'education_given_age' in self.learned_stats:
            if age_range in self.learned_stats['education_given_age']:
                edu_dist = self.learned_stats['education_given_age'][age_range]
                suggested_education = random.choices(
                    edu_dist['top_educations'],
                    weights=edu_dist['probabilities']
                )[0]
            else:
                suggested_education = "HS-grad"
        else:
            suggested_education = "HS-grad"
        
        suggested_num = EDUCATION_MAPPING.get(suggested_education, 9)
        
        return f"""生成教育字段（年龄{age}岁）:
- education: 建议 "{suggested_education}"（基于{age}岁人群的统计分布）
- education.num: {suggested_num}

注意：education和education.num必须严格对应！

输出JSON对象:"""

    def _get_work_prompt(self, condition: GenerationCondition, sample: Dict) -> str:
        education = sample.get("education", "HS-grad")
        education_num = sample.get("education.num", 9)
        
        # 【引导式生成】从P(occupation|education)和P(hours|education)采样
        edu_level = "low" if education_num <= 8 else ("medium" if education_num <= 12 else "high")
        
        if self.learned_stats and 'occupation_given_education' in self.learned_stats:
            if edu_level in self.learned_stats['occupation_given_education']:
                occ_dist = self.learned_stats['occupation_given_education'][edu_level]
                suggested_occupation = random.choices(
                    occ_dist['top_occupations'],
                    weights=occ_dist['probabilities']
                )[0]
            else:
                suggested_occupation = "Sales"
        else:
            suggested_occupation = "Sales"
        
        if self.learned_stats and 'hours_given_education' in self.learned_stats:
            if edu_level in self.learned_stats['hours_given_education']:
                hours_dist = self.learned_stats['hours_given_education'][edu_level]
                suggested_hours = int(np.clip(np.random.normal(hours_dist['mean'], hours_dist['std']), 1, 99))
            else:
                suggested_hours = 40
        else:
            suggested_hours = 40
        
        return f"""生成工作字段（教育:{education}）:
- workclass: Private占75%
- occupation: 建议 "{suggested_occupation}"（基于{education}的统计分布）
- hours.per.week: 建议 {suggested_hours} 小时

输出JSON对象:"""

    def _get_family_prompt(self, sample: Dict) -> str:
        age = sample.get("age", 30)
        sex = sample.get("sex", "Male")
        age_range = "young" if age <= 30 else ("middle" if age <= 55 else "senior")
        
        # 【引导式生成】从P(marital|age)和P(relationship|marital,sex)采样
        if self.learned_stats and 'marital_given_age' in self.learned_stats:
            if age_range in self.learned_stats['marital_given_age']:
                marital_dist = self.learned_stats['marital_given_age'][age_range]
                suggested_marital = random.choices(
                    marital_dist['top_status'],
                    weights=marital_dist['probabilities']
                )[0]
            else:
                suggested_marital = "Never-married" if age < 25 else "Married-civ-spouse"
        else:
            suggested_marital = "Never-married" if age < 25 else "Married-civ-spouse"
        
        # 根据婚姻和性别确定关系
        if suggested_marital == "Married-civ-spouse":
            suggested_relationship = "Husband" if sex == "Male" else "Wife"
        elif suggested_marital == "Never-married":
            suggested_relationship = "Not-in-family"
        else:
            suggested_relationship = "Not-in-family"
        
        return f"""生成家庭关系字段（年龄:{age}, 性别:{sex}）:
- marital.status: 建议 "{suggested_marital}"（基于{age}岁人群）
- relationship: 建议 "{suggested_relationship}"（必须与婚姻和性别一致）

输出JSON对象:"""

    def _get_financial_prompt(self, sample: Dict) -> str:
        education = sample.get("education", "HS-grad")
        education_num = sample.get("education.num", 9)
        edu_level = "low" if education_num <= 8 else ("medium" if education_num <= 12 else "high")
        
        # 【引导式生成】从P(capital.gain|education)采样
        if self.learned_stats and 'capital_gain_given_education' in self.learned_stats:
            if edu_level in self.learned_stats['capital_gain_given_education']:
                gain_dist = self.learned_stats['capital_gain_given_education'][edu_level]
                prob_nonzero = gain_dist['probability_nonzero']
                
                if random.random() < prob_nonzero:
                    mean_gain = gain_dist['mean_when_nonzero']
                    std_gain = gain_dist['std_when_nonzero']
                    suggested_gain = int(np.clip(np.random.normal(mean_gain, std_gain), 0, 99999))
                else:
                    suggested_gain = 0
            else:
                suggested_gain = 0
        else:
            suggested_gain = 0
        
        suggested_loss = 0 if random.random() < 0.95 else random.randint(100, 4356)
        suggested_fnlwgt = int(np.clip(np.random.normal(190000, 100000), 12285, 1484705))
        
        return f"""生成财务字段:
- capital.gain: 建议 {suggested_gain}（基于{education}人群）
- capital.loss: 建议 {suggested_loss}
- fnlwgt: 建议 {suggested_fnlwgt}

输出JSON对象:"""

    def _get_outcome_prompt(self, condition: GenerationCondition, sample: Dict) -> str:
        education = sample.get("education", "")
        hours = sample.get("hours.per.week", 40)
        occupation = sample.get("occupation", "")
        capital_gain = sample.get("capital.gain", 0)
        
        # 【引导式生成】从P(income|education, hours, occupation)采样
        if self.learned_stats and 'income_given_education' in self.learned_stats:
            if education in self.learned_stats['income_given_education']:
                income_dist = self.learned_stats['income_given_education'][education]
                prob_high_income = income_dist['>50K']
                
                # 根据其他因素调整概率
                if hours >= 45:
                    prob_high_income *= 1.2
                if occupation in ["Exec-managerial", "Prof-specialty"]:
                    prob_high_income *= 1.3
                if capital_gain > 5000:
                    prob_high_income *= 1.5
                
                prob_high_income = min(0.95, prob_high_income)
                suggested_income = ">50K" if random.random() < prob_high_income else "<=50K"
            else:
                suggested_income = "<=50K"
        else:
            suggested_income = "<=50K"
        
        # 如果有条件约束，使用条件
        if condition.income_class:
            suggested_income = condition.income_class
        
        return f"""生成收入类别（教育:{education}, 工时:{hours}h, 职业:{occupation}）:
- income: 建议 "{suggested_income}"（基于统计概率计算）

输出JSON对象:"""

    def _fallback_generate(self, group_name: str, condition: GenerationCondition, sample: Dict) -> Dict:
        """基于条件分布的回退生成（使用学到的统计特征）"""
        
        if group_name == "demographics":
            # 使用学到的年龄分布
            if self.learned_stats and 'age' in self.learned_stats:
                age_mean = self.learned_stats['age']['mean']
                age_std = self.learned_stats['age']['std']
                age_min = self.learned_stats['age']['min']
                age_max = self.learned_stats['age']['max']
                
                # 根据条件调整均值
                if condition.age_range == "young":
                    age_mean = 24
                elif condition.age_range == "middle":
                    age_mean = 42
                elif condition.age_range == "senior":
                    age_mean = 65
                
                age = int(np.clip(np.random.normal(age_mean, age_std), age_min, age_max))
            else:
                age_ranges = {"young": (17, 30), "middle": (31, 55), "senior": (56, 90)}
                age_range = age_ranges.get(condition.age_range, (17, 90))
                age = random.randint(*age_range)
            
            return {
                "age": age,
                "sex": condition.gender or random.choice(["Male", "Female"]),
                "race": random.choices(
                    ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                    weights=[85, 8, 4, 2, 1]
                )[0],
                "native.country": random.choices(
                    ["United-States", "Mexico", "Philippines", "Germany", "?"],
                    weights=[90, 3, 2, 2, 3]
                )[0]
            }
        
        elif group_name == "education":
            # 【条件分布采样】使用P(education|age)
            age = sample.get("age", 30)
            age_range = "young" if age <= 30 else ("middle" if age <= 55 else "senior")
            
            if self.learned_stats and 'education_given_age' in self.learned_stats:
                if age_range in self.learned_stats['education_given_age']:
                    edu_dist = self.learned_stats['education_given_age'][age_range]
                    education = random.choices(
                        edu_dist['top_educations'],
                        weights=edu_dist['probabilities']
                    )[0]
                else:
                    # 回退到条件约束
                    edu_levels = {
                        "low": ["7th-8th", "9th", "10th", "11th", "12th"],
                        "medium": ["HS-grad", "Some-college", "Assoc-voc"],
                        "high": ["Bachelors", "Masters", "Prof-school", "Doctorate"]
                    }
                    education = random.choice(edu_levels.get(condition.education_level, 
                                             ["HS-grad", "Some-college", "Bachelors"]))
            else:
                edu_levels = {
                    "low": ["7th-8th", "9th", "10th", "11th", "12th"],
                    "medium": ["HS-grad", "Some-college", "Assoc-voc"],
                    "high": ["Bachelors", "Masters", "Prof-school", "Doctorate"]
                }
                education = random.choice(edu_levels.get(condition.education_level, 
                                         ["HS-grad", "Some-college", "Bachelors"]))
            
            return {
                "education": education,
                "education.num": EDUCATION_MAPPING.get(education, 9)
            }
        
        elif group_name == "work":
            education = sample.get("education", "HS-grad")
            education_num = sample.get("education.num", 9)
            
            # 【条件分布采样】P(occupation|education)
            edu_level = "low" if education_num <= 8 else ("medium" if education_num <= 12 else "high")
            
            if self.learned_stats and 'occupation_given_education' in self.learned_stats:
                if edu_level in self.learned_stats['occupation_given_education']:
                    occ_dist = self.learned_stats['occupation_given_education'][edu_level]
                    occupation = random.choices(
                        occ_dist['top_occupations'],
                        weights=occ_dist['probabilities']
                    )[0]
                else:
                    occupation = random.choice(["Sales", "Craft-repair", "Other-service"])
            else:
                if education in ["Doctorate", "Prof-school", "Masters"]:
                    occupation = random.choice(["Prof-specialty", "Exec-managerial"])
                else:
                    occupation = random.choice(["Sales", "Craft-repair", "Other-service", "Adm-clerical"])
            
            # 【条件分布采样】P(hours|education)
            if self.learned_stats and 'hours_given_education' in self.learned_stats:
                if edu_level in self.learned_stats['hours_given_education']:
                    hours_dist = self.learned_stats['hours_given_education'][edu_level]
                    hours = int(np.clip(np.random.normal(hours_dist['mean'], hours_dist['std']), 1, 99))
                else:
                    hours = int(np.random.normal(40, 12))
            else:
                hours = int(np.random.normal(40, 12))
            
            return {
                "workclass": random.choices(
                    ["Private", "Self-emp-not-inc", "Federal-gov", "Local-gov", "?"],
                    weights=[75, 10, 5, 5, 5]
                )[0],
                "occupation": occupation,
                "hours.per.week": hours
            }
        
        elif group_name == "family":
            age = sample.get("age", 30)
            sex = sample.get("sex", "Male")
            
            # 【条件分布采样】P(marital|age)
            age_range = "young" if age <= 30 else ("middle" if age <= 55 else "senior")
            
            if self.learned_stats and 'marital_given_age' in self.learned_stats:
                if age_range in self.learned_stats['marital_given_age']:
                    marital_dist = self.learned_stats['marital_given_age'][age_range]
                    marital = random.choices(
                        marital_dist['top_status'],
                        weights=marital_dist['probabilities']
                    )[0]
                else:
                    marital = "Never-married" if age < 25 else random.choice(["Married-civ-spouse", "Divorced"])
            else:
                if age < 25:
                    marital = "Never-married"
                elif age < 60:
                    marital = random.choice(["Married-civ-spouse", "Never-married", "Divorced"])
                else:
                    marital = random.choice(["Married-civ-spouse", "Widowed", "Divorced"])
            
            # 【条件分布采样】P(relationship|marital,sex)
            relationship_key = f"{marital}_{sex}"
            if self.learned_stats and 'relationship_given_marital_sex' in self.learned_stats:
                if relationship_key in self.learned_stats['relationship_given_marital_sex']:
                    rel_dist = self.learned_stats['relationship_given_marital_sex'][relationship_key]
                    relationship = random.choices(
                        rel_dist['top_relationships'],
                        weights=rel_dist['probabilities']
                    )[0]
                else:
                    # 回退到规则
                    if marital == "Married-civ-spouse":
                        relationship = "Husband" if sex == "Male" else "Wife"
                    elif marital == "Never-married":
                        relationship = random.choice(["Not-in-family", "Own-child"])
                    else:
                        relationship = random.choice(["Not-in-family", "Unmarried"])
            else:
                if marital == "Married-civ-spouse":
                    relationship = "Husband" if sex == "Male" else "Wife"
                elif marital == "Never-married":
                    relationship = random.choice(["Not-in-family", "Own-child"])
                else:
                    relationship = random.choice(["Not-in-family", "Unmarried"])
            
            return {
                "marital.status": marital,
                "relationship": relationship
            }
        
        elif group_name == "financial":
            education = sample.get("education", "HS-grad")
            education_num = sample.get("education.num", 9)
            
            # 【条件分布采样】P(capital.gain|education)
            edu_level = "low" if education_num <= 8 else ("medium" if education_num <= 12 else "high")
            
            if self.learned_stats and 'capital_gain_given_education' in self.learned_stats:
                if edu_level in self.learned_stats['capital_gain_given_education']:
                    gain_dist = self.learned_stats['capital_gain_given_education'][edu_level]
                    prob_nonzero = gain_dist['probability_nonzero']
                    
                    if random.random() < prob_nonzero:
                        # 有资本收益，从条件分布采样
                        mean_gain = gain_dist['mean_when_nonzero']
                        std_gain = gain_dist['std_when_nonzero']
                        capital_gain = int(np.clip(np.random.normal(mean_gain, std_gain), 0, 99999))
                    else:
                        capital_gain = 0
                else:
                    capital_gain = 0 if random.random() < 0.9 else random.randint(100, 5000)
            else:
                # 回退到规则
                if education in ["Doctorate", "Prof-school", "Masters"] and random.random() < 0.3:
                    capital_gain = random.choice([0, 5178, 7298, 15024, 99999])
                else:
                    capital_gain = 0 if random.random() < 0.9 else random.randint(100, 5000)
            
            capital_loss = 0 if random.random() < 0.95 else random.randint(100, 4356)
            
            return {
                "capital.gain": capital_gain,
                "capital.loss": capital_loss,
                "fnlwgt": int(np.clip(np.random.normal(190000, 100000), 12285, 1484705))
            }
        
        elif group_name == "outcome":
            education = sample.get("education", "")
            hours = sample.get("hours.per.week", 40)
            occupation = sample.get("occupation", "")
            capital_gain = sample.get("capital.gain", 0)
            
            # 【条件分布采样】P(income|education)
            if self.learned_stats and 'income_given_education' in self.learned_stats:
                if education in self.learned_stats['income_given_education']:
                    income_dist = self.learned_stats['income_given_education'][education]
                    prob_high_income = income_dist['>50K']
                    
                    # 根据其他因素调整概率
                    if hours >= 45:
                        prob_high_income *= 1.2
                    if occupation in ["Exec-managerial", "Prof-specialty"]:
                        prob_high_income *= 1.3
                    if capital_gain > 5000:
                        prob_high_income *= 1.5
                    
                    prob_high_income = min(0.95, prob_high_income)  # 不超过95%
                    
                    income = ">50K" if random.random() < prob_high_income else "<=50K"
                else:
                    # 回退到评分法
                    score = 0
                    if education in ["Doctorate", "Prof-school", "Masters", "Bachelors"]:
                        score += 2
                    if hours >= 45:
                        score += 1
                    if occupation in ["Exec-managerial", "Prof-specialty"]:
                        score += 1
                    if capital_gain > 5000:
                        score += 2
                    income = ">50K" if score >= 3 else "<=50K"
            else:
                # 回退到评分法
                score = 0
                if education in ["Doctorate", "Prof-school", "Masters", "Bachelors"]:
                    score += 2
                if hours >= 45:
                    score += 1
                if occupation in ["Exec-managerial", "Prof-specialty"]:
                    score += 1
                if capital_gain > 5000:
                    score += 2
                income = ">50K" if score >= 3 else "<=50K"
            
            # 如果有条件约束，优先使用条件
            if condition.income_class:
                income = condition.income_class
            
            return {"income": income}
        
        return {}


# ============================================================================
#                      II. VERIFICATION - 验证阶段
# ============================================================================

class RuleBasedVerifier:
    """基于规则的验证器"""
    
    def verify(self, sample: Dict) -> Tuple[bool, List[str]]:
        """验证样本是否符合业务规则
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 1. 年龄范围检查
        try:
            age = int(sample.get("age", 0))
            if not (17 <= age <= 90):
                errors.append(f"年龄{age}超出合理范围17-90")
        except:
            errors.append("年龄字段无效")
        
        # 2. 教育一致性检查
        education = sample.get("education", "")
        education_num = sample.get("education.num", 0)
        expected_num = EDUCATION_MAPPING.get(education)
        if expected_num and expected_num != education_num:
            errors.append(f"教育程度{education}与年限{education_num}不匹配，应为{expected_num}")
        
        # 3. 婚姻-关系-性别一致性
        marital = sample.get("marital.status", "")
        relationship = sample.get("relationship", "")
        sex = sample.get("sex", "")
        
        if marital == "Married-civ-spouse":
            if sex == "Male" and relationship not in ["Husband"]:
                errors.append(f"已婚男性应为Husband，当前为{relationship}")
            elif sex == "Female" and relationship not in ["Wife"]:
                errors.append(f"已婚女性应为Wife，当前为{relationship}")
        
        # 4. 工作时长检查
        try:
            hours = int(sample.get("hours.per.week", 0))
            if not (1 <= hours <= 99):
                errors.append(f"每周工时{hours}超出范围1-99")
        except:
            errors.append("工作时长字段无效")
        
        # 5. 资本收益/损失检查
        try:
            capital_gain = int(sample.get("capital.gain", 0))
            capital_loss = int(sample.get("capital.loss", 0))
            if capital_gain < 0 or capital_gain > 99999:
                errors.append(f"资本收益{capital_gain}超出范围0-99999")
            if capital_loss < 0 or capital_loss > 4356:
                errors.append(f"资本损失{capital_loss}超出范围0-4356")
        except:
            errors.append("资本收益/损失字段无效")
        
        return len(errors) == 0, errors


# ============================================================================
#                      III. MAIN GENERATOR
# ============================================================================

class AdultDataGenerator:
    """Adult Census数据生成器主类"""
    
    def __init__(self, 
                 sample_file: Optional[str] = None,
                 use_heuristic: bool = True,
                 verbose: bool = True):
        """
        初始化生成器
        
        Args:
            sample_file: 真实样本CSV文件路径
            use_heuristic: 是否使用启发式示例选择
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.demo_manager = DemonstrationManager(use_heuristic=use_heuristic)
        self.verifier = RuleBasedVerifier()
        
        # 学习统计信息
        self.learned_stats = {}
        if sample_file:
            self._learn_from_samples(sample_file)
        
        self.decomposer = SampleWiseDecomposer(learned_stats=self.learned_stats)
        
        if self.verbose:
            print("[Adult Data Generator] 初始化完成")
    
    def _learn_from_samples(self, sample_file: str):
        """从真实样本中学习统计信息（包括条件分布）"""
        try:
            df = pd.read_csv(sample_file)
            
            # 加载示例
            self.demo_manager.load_samples_from_file(sample_file)
            
            # 1. 边缘分布
            self.learned_stats = {
                'age': {
                    'mean': df['age'].mean(),
                    'std': df['age'].std(),
                    'min': int(df['age'].min()),
                    'max': int(df['age'].max())
                },
                'hours': {
                    'mean': df['hours.per.week'].mean(),
                    'std': df['hours.per.week'].std()
                },
                'income_distribution': {
                    '<=50K': (df['income'] == '<=50K').sum() / len(df),
                    '>50K': (df['income'] == '>50K').sum() / len(df)
                }
            }
            
            # 2. 条件分布：教育 → 收入 P(income|education)
            self.learned_stats['income_given_education'] = {}
            for edu in df['education'].unique():
                edu_df = df[df['education'] == edu]
                if len(edu_df) > 0:
                    self.learned_stats['income_given_education'][edu] = {
                        '>50K': (edu_df['income'] == '>50K').sum() / len(edu_df),
                        '<=50K': (edu_df['income'] == '<=50K').sum() / len(edu_df)
                    }
            
            # 3. 条件分布：教育 → 职业 P(occupation|education)
            self.learned_stats['occupation_given_education'] = {}
            edu_levels = {
                'low': df[df['education.num'] <= 8],
                'medium': df[(df['education.num'] >= 9) & (df['education.num'] <= 12)],
                'high': df[df['education.num'] >= 13]
            }
            for level, level_df in edu_levels.items():
                occ_counts = level_df['occupation'].value_counts()
                self.learned_stats['occupation_given_education'][level] = {
                    'top_occupations': occ_counts.head(5).index.tolist(),
                    'probabilities': (occ_counts.head(5) / occ_counts.head(5).sum()).tolist()
                }
            
            # 4. 条件分布：教育 → 工作时长 (hours|education)
            self.learned_stats['hours_given_education'] = {}
            for level, level_df in edu_levels.items():
                self.learned_stats['hours_given_education'][level] = {
                    'mean': level_df['hours.per.week'].mean(),
                    'std': level_df['hours.per.week'].std()
                }
            
            # 5. 条件分布：年龄 → 教育 (education|age)
            age_groups = {
                'young': df[df['age'] <= 30],
                'middle': df[(df['age'] > 30) & (df['age'] <= 55)],
                'senior': df[df['age'] > 55]
            }
            self.learned_stats['education_given_age'] = {}
            for age_range, age_df in age_groups.items():
                edu_counts = age_df['education'].value_counts()
                self.learned_stats['education_given_age'][age_range] = {
                    'top_educations': edu_counts.head(5).index.tolist(),
                    'probabilities': (edu_counts.head(5) / edu_counts.head(5).sum()).tolist()
                }
            
            # 6. 条件分布：年龄+教育 → 婚姻状况
            self.learned_stats['marital_given_age'] = {}
            for age_range, age_df in age_groups.items():
                marital_counts = age_df['marital.status'].value_counts()
                self.learned_stats['marital_given_age'][age_range] = {
                    'top_status': marital_counts.head(3).index.tolist(),
                    'probabilities': (marital_counts.head(3) / marital_counts.head(3).sum()).tolist()
                }
            
            # 7. 条件分布：教育+工时 → 资本收益 (capital.gain|education,hours)
            self.learned_stats['capital_gain_given_education'] = {}
            for level, level_df in edu_levels.items():
                has_gain = level_df[level_df['capital.gain'] > 0]
                self.learned_stats['capital_gain_given_education'][level] = {
                    'probability_nonzero': len(has_gain) / len(level_df) if len(level_df) > 0 else 0,
                    'mean_when_nonzero': has_gain['capital.gain'].mean() if len(has_gain) > 0 else 0,
                    'std_when_nonzero': has_gain['capital.gain'].std() if len(has_gain) > 0 else 0
                }
            
            # 8. 条件分布：性别+婚姻 → 家庭关系
            self.learned_stats['relationship_given_marital_sex'] = {}
            for marital in df['marital.status'].unique():
                for sex in ['Male', 'Female']:
                    subset = df[(df['marital.status'] == marital) & (df['sex'] == sex)]
                    if len(subset) > 0:
                        rel_counts = subset['relationship'].value_counts()
                        key = f"{marital}_{sex}"
                        self.learned_stats['relationship_given_marital_sex'][key] = {
                            'top_relationships': rel_counts.head(2).index.tolist(),
                            'probabilities': (rel_counts.head(2) / rel_counts.head(2).sum()).tolist()
                        }
            
            # 9. 相关系数计算
            numeric_cols = ['age', 'education.num', 'hours.per.week', 'capital.gain']
            self.learned_stats['correlations'] = {}
            for col1 in numeric_cols:
                for col2 in numeric_cols:
                    if col1 != col2:
                        corr = df[col1].corr(df[col2])
                        self.learned_stats['correlations'][f"{col1}_{col2}"] = corr
            
            if self.verbose:
                print(f"  [条件分布学习] 完成，样本数: {len(df)}")
                print(f"    - 平均年龄: {self.learned_stats['age']['mean']:.1f} ± {self.learned_stats['age']['std']:.1f}")
                print(f"    - 平均工时: {self.learned_stats['hours']['mean']:.1f}")
                print(f"    - 高收入比例: {self.learned_stats['income_distribution']['>50K']:.2%}")
                print(f"    - 学习了 {len(self.learned_stats['income_given_education'])} 个教育等级的收入条件分布")
                print(f"    - 学习了 {len(self.learned_stats['occupation_given_education'])} 个教育水平的职业条件分布")
                
                # 示例：显示高学历人群的高收入概率
                if 'Doctorate' in self.learned_stats['income_given_education']:
                    doctorate_high_income = self.learned_stats['income_given_education']['Doctorate']['>50K']
                    print(f"    - 博士学位高收入概率: {doctorate_high_income:.1%}")
        
        except Exception as e:
            print(f"  [警告] 学习统计信息失败: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_batch(self,
                      n_samples: int,
                      condition: Optional[GenerationCondition] = None,
                      max_retries: int = 3) -> List[Dict]:
        """批量生成样本
        
        Args:
            n_samples: 生成样本数
            condition: 生成条件
            max_retries: 每个样本最大重试次数
        
        Returns:
            生成的样本列表
        """
        if condition is None:
            condition = GenerationCondition()
        
        samples = []
        self.decomposer.reset_cache()
        
        for i in range(n_samples):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  [生成进度] {i + 1}/{n_samples}")
            
            sample = None
            for retry in range(max_retries):
                try:
                    sample = self.decomposer.decompose_and_generate(condition)
                    
                    # 验证
                    is_valid, errors = self.verifier.verify(sample)
                    if is_valid:
                        break
                    elif self.verbose and retry == max_retries - 1:
                        print(f"  [警告] 样本{i+1}验证失败: {errors[:2]}")
                except Exception as e:
                    if self.verbose and retry == max_retries - 1:
                        print(f"  [错误] 样本{i+1}生成失败: {e}")
            
            if sample:
                samples.append(sample)
                self.decomposer.update_cache(sample)
        
        if self.verbose:
            print(f"  [完成] 成功生成 {len(samples)}/{n_samples} 个样本")
        
        return samples
    
    def save_to_csv(self, samples: List[Dict], output_file: str):
        """保存样本到CSV文件"""
        df = pd.DataFrame(samples)
        
        # 确保列顺序与原始数据一致
        column_order = [
            'age', 'workclass', 'fnlwgt', 'education', 'education.num',
            'marital.status', 'occupation', 'relationship', 'race', 'sex',
            'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income'
        ]
        
        # 只保留存在的列
        existing_cols = [col for col in column_order if col in df.columns]
        df = df[existing_cols]
        
        df.to_csv(output_file, index=False)
        
        if self.verbose:
            print(f"  [保存] 已保存到 {output_file}")


# ============================================================================
#                      IV. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("Adult Census Synthetic Data Generator")
    print("=" * 60)
    
    # 初始化生成器（使用真实样本学习）
    generator = AdultDataGenerator(
        sample_file="archive/adult.csv",
        use_heuristic=True,
        verbose=True
    )
    
    # 生成不同条件下的样本
    print("\n[任务1] 生成高收入、高教育人群样本 (50个)")
    high_income_condition = GenerationCondition(
        age_range="middle",
        education_level="high",
        income_class=">50K"
    )
    high_income_samples = generator.generate_batch(50, high_income_condition)
    generator.save_to_csv(high_income_samples, "synthetic_adult_high_income.csv")
    
    print("\n[任务2] 生成年轻低收入人群样本 (50个)")
    young_condition = GenerationCondition(
        age_range="young",
        education_level="medium",
        income_class="<=50K"
    )
    young_samples = generator.generate_batch(50, young_condition)
    generator.save_to_csv(young_samples, "synthetic_adult_young.csv")
    
    print("\n[任务3] 生成无条件随机样本 (100个)")
    random_samples = generator.generate_batch(100)
    generator.save_to_csv(random_samples, "synthetic_adult_random.csv")
    
    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)
