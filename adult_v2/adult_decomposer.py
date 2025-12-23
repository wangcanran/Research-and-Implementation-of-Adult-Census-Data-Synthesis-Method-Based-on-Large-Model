"""
Adult Census Data - Sample-Wise Decomposer
样本分解器 - 分步生成 + 引导式Prompt
"""

import json
import random
import numpy as np
from typing import Dict, Optional
from openai import OpenAI

from .adult_task_spec import (
    GenerationCondition, FIELD_GROUPS, EDUCATION_MAPPING,
    EDUCATION_NUM_TO_NAME, validate_sample
)

# Import config
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from adult_config import OPENAI_API_KEY, OPENAI_API_BASE, FIXED_MODEL_NAME

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
MODEL_NAME = FIXED_MODEL_NAME


class AdultSampleWiseDecomposer:
    """
    样本分解器
    
    功能：
    1. 分步生成（demographics → education → work → family → financial → outcome）
    2. 引导式Prompt（在Prompt中嵌入条件分布采样的建议值）
    3. 条件依赖验证（LLM生成后验证，不符合则fallback）
    4. 记忆机制（缓存已生成样本的统计信息）
    """
    
    def __init__(self, learned_stats: Dict = None):
        """
        Args:
            learned_stats: 从真实数据学习的统计信息（条件分布）
        """
        self.learned_stats = learned_stats if learned_stats is not None else {}
        
        # 记忆缓存：记录本批次已生成样本的统计信息
        self.generation_cache = {
            'ages': [],
            'hours': [],
            'educations': [],
            'incomes': [],
            'count': 0
        }
    
    def update_cache(self, sample: Dict):
        """更新生成缓存（每生成一个样本后调用）"""
        try:
            if 'age' in sample:
                self.generation_cache['ages'].append(int(sample['age']))
            if 'hours.per.week' in sample:
                self.generation_cache['hours'].append(int(sample['hours.per.week']))
            if 'education.num' in sample:
                self.generation_cache['educations'].append(int(sample['education.num']))
            if 'income' in sample:
                self.generation_cache['incomes'].append(sample['income'])
            self.generation_cache['count'] += 1
        except:
            pass
    
    def reset_cache(self):
        """重置缓存（开始新一批生成时调用）"""
        self.generation_cache = {
            'ages': [],
            'hours': [],
            'educations': [],
            'incomes': [],
            'count': 0
        }
    
    def get_cache_stats(self) -> Dict:
        """获取当前缓存的统计信息"""
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
        """生成单个字段组（使用LLM + 引导式Prompt）"""
        condition = context["condition"]
        
        # 获取引导式Prompt（内嵌条件分布采样的建议值）
        prompts = {
            "demographics": self._get_demographics_prompt(condition),
            "education": self._get_education_prompt(condition, current_sample),
            "work": self._get_work_prompt(condition, current_sample),
            "family": self._get_family_prompt(current_sample),
            "financial": self._get_financial_prompt(current_sample),
            "outcome": self._get_outcome_prompt(condition, current_sample)
        }
        
        # 调用LLM
        try:
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
            
            result = json.loads(content)
            
            # 后处理：确保education和education.num一致
            if group_name == "education" and "education" in result:
                edu_name = result["education"]
                if edu_name in EDUCATION_MAPPING:
                    result["education.num"] = EDUCATION_MAPPING[edu_name]
            
            # 条件依赖验证：检查LLM生成是否符合条件分布
            if not self._verify_conditional_dependency(group_name, result, current_sample, condition):
                # 如果不符合，使用fallback的条件分布采样
                return self._fallback_generate(group_name, condition, current_sample)
            
            return result
            
        except json.JSONDecodeError:
            return self._fallback_generate(group_name, condition, current_sample)
        except Exception as e:
            print(f"  [Decomposer] LLM生成失败({group_name}): {e}")
            return self._fallback_generate(group_name, condition, current_sample)
    
    def _verify_conditional_dependency(self, group_name: str, result: Dict,
                                       current_sample: Dict, condition: GenerationCondition) -> bool:
        """
        验证LLM生成的字段是否符合条件依赖关系
        
        Returns:
            True: 符合条件依赖，可以使用
            False: 不符合，需要回退到fallback
        """
        # 简化验证：主要检查关键依赖
        
        if group_name == "education":
            # 验证教育-年限映射
            education = result.get("education", "")
            education_num = result.get("education.num", 0)
            if education in EDUCATION_MAPPING:
                expected_num = EDUCATION_MAPPING[education]
                if education_num != expected_num:
                    return False
        
        elif group_name == "family":
            # 验证婚姻-性别-关系一致性
            marital = result.get("marital.status", "")
            relationship = result.get("relationship", "")
            sex = current_sample.get("sex", "")
            
            if marital == "Married-civ-spouse":
                if sex == "Male" and relationship != "Husband":
                    return False
                if sex == "Female" and relationship != "Wife":
                    return False
        
        return True
    
    def _clean_json(self, content: str) -> str:
        """清理JSON格式"""
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        return content.strip()
    
    # ========== 引导式Prompt生成方法 ==========
    
    def _get_demographics_prompt(self, condition: GenerationCondition) -> str:
        """生成demographics字段的引导式Prompt"""
        # 从条件分布采样得到建议值
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
- native.country: 原籍国家，United-States占90%

输出JSON对象:"""
    
    def _get_education_prompt(self, condition: GenerationCondition, sample: Dict) -> str:
        """生成education字段的引导式Prompt"""
        age = sample.get("age", 30)
        age_range = "young" if age <= 30 else ("middle" if age <= 55 else "senior")
        
        # 从P(education|age)采样建议值
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
        """生成work字段的引导式Prompt"""
        education = sample.get("education", "HS-grad")
        education_num = sample.get("education.num", 9)
        
        # 从P(occupation|education)和P(hours|education)采样
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
        """生成family字段的引导式Prompt"""
        age = sample.get("age", 30)
        sex = sample.get("sex", "Male")
        age_range = "young" if age <= 30 else ("middle" if age <= 55 else "senior")
        
        # 从P(marital|age)和P(relationship|marital,sex)采样
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
        """生成financial字段的引导式Prompt"""
        education = sample.get("education", "HS-grad")
        education_num = sample.get("education.num", 9)
        edu_level = "low" if education_num <= 8 else ("medium" if education_num <= 12 else "high")
        
        # 从P(capital.gain|education)采样
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
        """生成outcome字段的引导式Prompt"""
        education = sample.get("education", "")
        hours = sample.get("hours.per.week", 40)
        occupation = sample.get("occupation", "")
        capital_gain = sample.get("capital.gain", 0)
        
        # 从P(income|education, hours, occupation)采样
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
        """基于条件分布的回退生成（纯统计采样）"""
        # 这里使用简化的fallback，实际可以更完善
        # 直接返回从条件分布采样的结果
        
        if group_name == "demographics":
            age_range = {"young": (17, 30), "middle": (31, 55), "senior": (56, 90)}
            age_r = age_range.get(condition.age_range, (17, 90))
            return {
                "age": random.randint(*age_r),
                "sex": condition.gender or random.choice(["Male", "Female"]),
                "race": random.choices(
                    ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                    weights=[85, 8, 4, 2, 1]
                )[0],
                "native.country": "United-States"
            }
        
        elif group_name == "education":
            return {
                "education": "HS-grad",
                "education.num": 9
            }
        
        elif group_name == "work":
            return {
                "workclass": "Private",
                "occupation": "Sales",
                "hours.per.week": 40
            }
        
        elif group_name == "family":
            sex = sample.get("sex", "Male")
            age = sample.get("age", 30)
            marital = "Married-civ-spouse" if age >= 30 else "Never-married"
            relationship = "Husband" if (sex == "Male" and marital == "Married-civ-spouse") else (
                "Wife" if (sex == "Female" and marital == "Married-civ-spouse") else "Not-in-family"
            )
            return {
                "marital.status": marital,
                "relationship": relationship
            }
        
        elif group_name == "financial":
            return {
                "capital.gain": 0,
                "capital.loss": 0,
                "fnlwgt": int(np.random.normal(190000, 100000))
            }
        
        elif group_name == "outcome":
            return {
                "income": condition.income_class or "<=50K"
            }
        
        return {}
