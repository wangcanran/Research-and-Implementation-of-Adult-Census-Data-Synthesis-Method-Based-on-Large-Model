"""
Sample-Wise Decomposer Module
从 data_generator.py 提取，修改为 Adult Census
最核心的模块：分字段组生成样本
"""

import json
import random
from typing import Dict, List
import numpy as np

from .config import client, MODEL_NAME
from .task_spec import TASK_SPECIFICATION, FIELD_GROUPS, GenerationCondition, EDUCATION_MAPPING, EDUCATION_NUM_TO_NAME
from .demonstration_manager import DemonstrationManager


class SampleWiseDecomposer:
    """
    样本级分解器（照搬data_generator.py的核心策略）
    
    核心功能：
    1. 分字段组生成（demographics → education → work → family → financial → outcome）
    2. 基于条件分布引导生成
    3. JSON解析和验证
    4. 回退生成机制
    """
    
    def __init__(self, demo_manager: DemonstrationManager, target_distribution: Dict = None):
        """
        Args:
            demo_manager: 示例管理器
            target_distribution: 目标分布（用于自适应校正）
        """
        self.demo_manager = demo_manager
        self.target_distribution = target_distribution or {}
        
        # 生成缓存（照搬data_generator.py）
        self.generation_cache = {
            'ages': [],
            'hours': [],
            'educations': [],
            'incomes': [],
            'count': 0
        }
    
    def update_cache(self, sample: Dict):
        """更新生成缓存（照搬）"""
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
        """重置缓存"""
        self.generation_cache = {
            'ages': [],
            'hours': [],
            'educations': [],
            'incomes': [],
            'count': 0
        }
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        ages = self.generation_cache.get('ages', [])
        if not ages:
            return {}
        
        stats = {
            'age_mean': int(np.mean(ages)),
            'age_std': int(np.std(ages)),
            'count': self.generation_cache['count']
        }
        
        hours = self.generation_cache.get('hours', [])
        if hours:
            stats['hours_mean'] = int(np.mean(hours))
        
        incomes = self.generation_cache.get('incomes', [])
        if incomes:
            high_income_count = sum(1 for inc in incomes if inc == '>50K')
            stats['high_income_ratio'] = high_income_count / len(incomes)
        
        return stats
    
    def _get_adaptive_hint(self, field: str, current_sample: Dict) -> str:
        """生成自适应引导提示（核心改进）"""
        cache_stats = self.get_cache_stats()
        
        # 需要至少5个样本才启用
        if not cache_stats or cache_stats['count'] < 5:
            return ""
        
        hints = []
        
        # 收入分布校正
        if field == 'outcome' and 'income' in self.target_distribution:
            current_ratio = cache_stats.get('high_income_ratio', 0)
            target_ratio = self.target_distribution['income'].get('>50K', 0.24)
            
            deviation = abs(current_ratio - target_ratio)
            if deviation > 0.1:  # 偏离>10%
                if current_ratio < target_ratio:
                    hints.append(f"当前高收入比例{current_ratio*100:.1f}%偏低，目标{target_ratio*100:.1f}%，建议多生成>50K")
                else:
                    hints.append(f"当前高收入比例{current_ratio*100:.1f}%偏高，目标{target_ratio*100:.1f}%，建议多生成<=50K")
        
        # 年龄分布校正
        if field in ['demographics', 'work'] and cache_stats.get('age_mean'):
            current_age = cache_stats['age_mean']
            target_age = 38  # Adult数据集的平均年龄
            
            if abs(current_age - target_age) > 5:
                if current_age < target_age:
                    hints.append(f"当前平均年龄{current_age}岁偏低，建议生成较大年龄(35-50岁)")
                else:
                    hints.append(f"当前平均年龄{current_age}岁偏高，建议生成较小年龄(25-35岁)")
        
        if hints:
            return "\n【自适应校正】" + "; ".join(hints)
        return ""
    
    def decompose_and_generate(self, condition: GenerationCondition) -> Dict:
        """
        分步生成完整样本（照搬data_generator.py的核心逻辑）
        
        生成顺序：demographics → education → work → family → financial → outcome
        """
        sample = {}
        context = {"condition": condition}
        
        # 生成顺序（照搬）
        generation_order = ["demographics", "education", "work", "family", "financial", "outcome"]
        
        for group_name in generation_order:
            group_data = self._generate_field_group(group_name, sample, context)
            sample.update(group_data)
        
        return sample
    
    def _generate_field_group(self, group_name: str, current_sample: Dict, context: Dict) -> Dict:
        """生成单个字段组（照搬data_generator.py）"""
        condition = context["condition"]
        
        # 构建prompt
        prompt = self._build_prompt(group_name, current_sample, condition)
        
        # 调用LLM
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": TASK_SPECIFICATION},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析JSON
            result = self._parse_json_response(content)
            
            # 验证和清理
            result = self._validate_and_clean(result, group_name, current_sample)
            
            return result
        except Exception as e:
            # 回退生成（照搬data_generator.py）
            return self._fallback_generate(group_name, condition, current_sample)
    
    def _build_prompt(self, group_name: str, current_sample: Dict, condition: GenerationCondition) -> str:
        """构建prompt（照搬data_generator.py结构 + 自适应引导）"""
        # 选择示例
        demos = self.demo_manager.select_demonstrations(k=3, condition=condition)
        demo_text = self.demo_manager.format_demonstrations(demos)
        
        # 当前已生成字段
        current_fields = ""
        if current_sample:
            current_fields = "Current partial record:\n"
            current_fields += json.dumps(current_sample, indent=2) + "\n\n"
        
        # 条件约束
        constraints = ""
        if condition.income:
            constraints += f"- Target income: {condition.income}\n"
        if condition.age_range:
            constraints += f"- Age range: {condition.age_range}\n"
        if condition.education_level:
            constraints += f"- Education level: {condition.education_level}\n"
        
        # 【核心改进】自适应引导提示
        adaptive_hint = self._get_adaptive_hint(group_name, current_sample)
        
        # 组装prompt
        prompt = f"""{demo_text}
{current_fields}
Generate the following fields for an Adult Census record: {FIELD_GROUPS[group_name]}

{constraints}{adaptive_hint}

IMPORTANT: Generate DIVERSE values. Vary the choices meaningfully - different races, countries, workclasses, occupations, etc. Avoid repeating similar patterns.

Output ONLY a JSON object with these fields, no additional text:"""
        
        return prompt
    
    def _parse_json_response(self, content: str) -> Dict:
        """解析JSON响应（照搬data_generator.py）"""
        # 清理markdown code blocks
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        return json.loads(content)
    
    def _validate_and_clean(self, result: Dict, group_name: str, current_sample: Dict) -> Dict:
        """验证和清理（照搬data_generator.py）"""
        cleaned = {}
        expected_fields = FIELD_GROUPS[group_name]
        
        for field in expected_fields:
            if field in result:
                cleaned[field] = result[field]
        
        # 特殊处理education.num和education的一致性
        if 'education' in cleaned and 'education.num' not in cleaned:
            if cleaned['education'] in EDUCATION_MAPPING:
                cleaned['education.num'] = EDUCATION_MAPPING[cleaned['education']]
        
        if 'education.num' in cleaned and 'education' not in cleaned:
            edu_num = int(cleaned['education.num'])
            if edu_num in EDUCATION_NUM_TO_NAME:
                cleaned['education'] = EDUCATION_NUM_TO_NAME[edu_num]
        
        return cleaned
    
    def _fallback_generate(self, group_name: str, condition: GenerationCondition, sample: Dict) -> Dict:
        """回退生成（照搬data_generator.py的规则生成）"""
        result = {}
        
        if group_name == "demographics":
            # 随机生成demographics
            if condition.age_range:
                age_ranges = {'young': (17, 30), 'middle': (31, 55), 'senior': (56, 90)}
                min_age, max_age = age_ranges.get(condition.age_range, (17, 90))
                result['age'] = random.randint(min_age, max_age)
            else:
                result['age'] = random.randint(25, 60)
            
            result['sex'] = random.choice(['Male', 'Female'])
            result['race'] = random.choice(['White', 'Black', 'Asian-Pac-Islander', 'Other'])
            result['native.country'] = 'United-States'
        
        elif group_name == "education":
            result['education.num'] = random.randint(9, 13)
            result['education'] = EDUCATION_NUM_TO_NAME.get(result['education.num'], 'HS-grad')
        
        elif group_name == "work":
            result['workclass'] = random.choice(['Private', 'Self-emp-not-inc', 'Federal-gov'])
            result['occupation'] = random.choice(['Sales', 'Craft-repair', 'Exec-managerial'])
            result['hours.per.week'] = random.choice([40, 35, 45, 50])
        
        elif group_name == "family":
            age = sample.get('age', 40)
            if age < 25:
                result['marital.status'] = 'Never-married'
                result['relationship'] = random.choice(['Own-child', 'Not-in-family'])
            else:
                result['marital.status'] = random.choice(['Married-civ-spouse', 'Never-married', 'Divorced'])
                if result['marital.status'] == 'Married-civ-spouse':
                    sex = sample.get('sex', 'Male')
                    result['relationship'] = 'Husband' if sex == 'Male' else 'Wife'
                else:
                    result['relationship'] = 'Not-in-family'
        
        elif group_name == "financial":
            result['capital.gain'] = 0
            result['capital.loss'] = 0
            result['fnlwgt'] = random.randint(100000, 300000)
        
        elif group_name == "outcome":
            edu_num = sample.get('education.num', 9)
            hours = sample.get('hours.per.week', 40)
            
            # 简单规则
            if edu_num >= 13 and hours >= 40:
                result['income'] = '>50K' if random.random() > 0.3 else '<=50K'
            else:
                result['income'] = '<=50K' if random.random() > 0.2 else '>50K'
        
        return result
