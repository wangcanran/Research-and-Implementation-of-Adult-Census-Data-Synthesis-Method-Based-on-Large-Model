"""
Adult Census Data - Statistical Learner
统计学习器 - 从真实数据学习条件分布
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .adult_task_spec import EDUCATION_MAPPING


class AdultStatisticalLearner:
    """
    统计学习器
    
    功能：
    从真实Adult数据中学习：
    1. 边缘分布（age, hours, income等）
    2. 条件分布（9种关键条件分布）
    3. 相关系数矩阵
    """
    
    def __init__(self):
        self.learned_stats = {}
    
    def learn_from_data(self, file_path: str) -> Dict:
        """
        从CSV文件学习统计特征
        
        Returns:
            learned_stats: 包含各种统计信息的字典
        """
        print(f"\n[StatLearner] 开始学习统计特征...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"  加载数据: {len(df)} 条")
            
            # 清理数据
            df = df.replace('?', np.nan)
            
            # 1. 边缘分布
            self._learn_marginal_distributions(df)
            
            # 2. 条件分布
            self._learn_conditional_distributions(df)
            
            # 3. 相关系数
            self._learn_correlations(df)
            
            print(f"  学习完成，共 {len(self.learned_stats)} 个统计特征")
            
            return self.learned_stats
            
        except Exception as e:
            print(f"  [错误] 学习失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _learn_marginal_distributions(self, df: pd.DataFrame):
        """学习边缘分布"""
        # Age
        self.learned_stats['age'] = {
            'mean': float(df['age'].mean()),
            'std': float(df['age'].std()),
            'min': int(df['age'].min()),
            'max': int(df['age'].max())
        }
        
        # Hours
        self.learned_stats['hours'] = {
            'mean': float(df['hours.per.week'].mean()),
            'std': float(df['hours.per.week'].std())
        }
        
        # Income distribution
        income_counts = df['income'].value_counts()
        total = len(df)
        self.learned_stats['income_distribution'] = {
            '<=50K': float(income_counts.get('<=50K', 0) / total),
            '>50K': float(income_counts.get('>50K', 0) / total)
        }
        print(f"    [OK] Marginal distributions")
    
    def _learn_conditional_distributions(self, df: pd.DataFrame):
        """学习9种条件分布"""
        
        # 1. P(income|education)
        self.learned_stats['income_given_education'] = {}
        for edu in df['education'].unique():
            if pd.isna(edu):
                continue
            edu_df = df[df['education'] == edu]
            if len(edu_df) > 0:
                income_counts = edu_df['income'].value_counts()
                total = len(edu_df)
                self.learned_stats['income_given_education'][edu] = {
                    '>50K': float(income_counts.get('>50K', 0) / total),
                    '<=50K': float(income_counts.get('<=50K', 0) / total)
                }
        
        # 2. P(occupation|education)
        self.learned_stats['occupation_given_education'] = {}
        edu_levels = {
            'low': df[df['education.num'] <= 8],
            'medium': df[(df['education.num'] >= 9) & (df['education.num'] <= 12)],
            'high': df[df['education.num'] >= 13]
        }
        for level, level_df in edu_levels.items():
            occ_counts = level_df['occupation'].value_counts()
            top5 = occ_counts.head(5)
            self.learned_stats['occupation_given_education'][level] = {
                'top_occupations': top5.index.tolist(),
                'probabilities': (top5 / top5.sum()).tolist()
            }
        
        # 3. P(hours|education)
        self.learned_stats['hours_given_education'] = {}
        for level, level_df in edu_levels.items():
            self.learned_stats['hours_given_education'][level] = {
                'mean': float(level_df['hours.per.week'].mean()),
                'std': float(level_df['hours.per.week'].std())
            }
        
        # 4. P(education|age)
        age_groups = {
            'young': df[df['age'] <= 30],
            'middle': df[(df['age'] > 30) & (df['age'] <= 55)],
            'senior': df[df['age'] > 55]
        }
        self.learned_stats['education_given_age'] = {}
        for age_range, age_df in age_groups.items():
            edu_counts = age_df['education'].value_counts()
            top5 = edu_counts.head(5)
            self.learned_stats['education_given_age'][age_range] = {
                'top_educations': top5.index.tolist(),
                'probabilities': (top5 / top5.sum()).tolist()
            }
        
        # 5. P(marital|age)
        self.learned_stats['marital_given_age'] = {}
        for age_range, age_df in age_groups.items():
            marital_counts = age_df['marital.status'].value_counts()
            top3 = marital_counts.head(3)
            self.learned_stats['marital_given_age'][age_range] = {
                'top_status': top3.index.tolist(),
                'probabilities': (top3 / top3.sum()).tolist()
            }
        
        # 6. P(relationship|marital,sex)
        self.learned_stats['relationship_given_marital_sex'] = {}
        for marital in df['marital.status'].unique():
            if pd.isna(marital):
                continue
            for sex in ['Male', 'Female']:
                key = f"{marital}_{sex}"
                subset = df[(df['marital.status'] == marital) & (df['sex'] == sex)]
                if len(subset) > 0:
                    rel_counts = subset['relationship'].value_counts()
                    top3 = rel_counts.head(3)
                    self.learned_stats['relationship_given_marital_sex'][key] = {
                        'top_relationships': top3.index.tolist(),
                        'probabilities': (top3 / top3.sum()).tolist()
                    }
        
        # 7. P(capital.gain|education)
        self.learned_stats['capital_gain_given_education'] = {}
        for level, level_df in edu_levels.items():
            has_gain = level_df[level_df['capital.gain'] > 0]
            self.learned_stats['capital_gain_given_education'][level] = {
                'probability_nonzero': len(has_gain) / len(level_df) if len(level_df) > 0 else 0,
                'mean_when_nonzero': float(has_gain['capital.gain'].mean()) if len(has_gain) > 0 else 0,
                'std_when_nonzero': float(has_gain['capital.gain'].std()) if len(has_gain) > 0 else 0
            }
        
        print(f"    [OK] Conditional distributions (9 types)")
    
    def _learn_correlations(self, df: pd.DataFrame):
        """学习相关系数"""
        # 选择数值字段计算相关系数
        numerical_fields = ['age', 'education.num', 'hours.per.week', 'capital.gain', 'capital.loss']
        
        correlations = {}
        for i, field1 in enumerate(numerical_fields):
            for field2 in numerical_fields[i+1:]:
                corr = df[field1].corr(df[field2])
                if not pd.isna(corr):
                    correlations[f"{field1}_{field2}"] = float(corr)
        
        self.learned_stats['correlations'] = correlations
        print(f"    [OK] Correlations")
    
    def get_stats(self) -> Dict:
        """获取学习到的统计信息"""
        return self.learned_stats
    
    def save_stats(self, file_path: str):
        """保存统计信息到文件"""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.learned_stats, f, indent=2)
        print(f"  [StatLearner] 统计信息已保存到: {file_path}")
    
    def load_stats(self, file_path: str):
        """从文件加载统计信息"""
        import json
        with open(file_path, 'r') as f:
            self.learned_stats = json.load(f)
        print(f"  [StatLearner] 统计信息已加载: {len(self.learned_stats)} 个特征")
