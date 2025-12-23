"""
Task Specification Module
从 data_generator.py 提取，修改为 Adult Census 数据集
"""

from typing import Optional, Dict
from dataclasses import dataclass

# ============================================================================
#                      Task Specification (Adult Census)
# ============================================================================

TASK_SPECIFICATION = """
You are an expert in generating Adult Census survey data. Your task is to generate realistic census records that follow real-world demographic and socioeconomic patterns.

## Adult Census Data Schema

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| age | Integer | Age in years | 17-90 |
| workclass | String | Work class | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked |
| fnlwgt | Integer | Final weight | Census sampling weight (100000-500000) |
| education | String | Education level | Preschool, 1st-4th, ..., HS-grad, Some-college, Bachelors, Masters, Doctorate |
| education.num | Integer | Education years | 1-16 (maps to education level) |
| marital.status | String | Marital status | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse |
| occupation | String | Occupation | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces |
| relationship | String | Relationship | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried |
| race | String | Race | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black |
| sex | String | Sex | Female, Male |
| capital.gain | Integer | Capital gain | 0-99999 (90% are 0) |
| capital.loss | Integer | Capital loss | 0-4356 (90% are 0) |
| hours.per.week | Integer | Hours per week | 1-99 (typical: 40 for full-time) |
| native.country | String | Native country | United-States (90%), or 40+ other countries |
| income | String | Income class | <=50K (76%), >50K (24%) |

## Core Business Rules

1. **Age-Education Correlation**: 
   - Older generations tend to have lower education levels
   - Age < 25: typically HS-grad or Some-college
   - Age 25-50: Bachelors or higher more common
   - Age > 60: HS-grad or less more common

2. **Education-Income Correlation**:
   - Bachelors+ → higher chance of >50K income
   - HS-grad or less → typically <=50K income
   - Masters/Doctorate → 70%+ chance of >50K

3. **Work Hours-Income Correlation**:
   - 40+ hours/week → higher chance of >50K
   - Part-time (< 35 hours) → typically <=50K

4. **Marital-Relationship Consistency**:
   - Married-civ-spouse → Husband or Wife
   - Never-married → Not-in-family or Own-child
   - Divorced/Separated → Not-in-family

5. **Age-Marital Consistency**:
   - Age < 18: Never-married (no married)
   - Age < 25: mostly Never-married
   - Age 25-50: mostly Married-civ-spouse
   - Age > 60: higher Widowed/Divorced rate

6. **Capital Gain/Loss Distribution**:
   - 90% have capital.gain = 0
   - 90% have capital.loss = 0
   - If capital.gain > 0, typically 5000-99999
   - If capital.loss > 0, typically 1500-4000

7. **Occupation-Education Consistency**:
   - Prof-specialty, Exec-managerial → requires Bachelors+
   - Tech-support → Some-college or higher
   - Handlers-cleaners, Farming-fishing → typically HS-grad or less

8. **Sex-Occupation Patterns** (historical data, 1994):
   - Exec-managerial, Tech-support → more Male
   - Adm-clerical, Other-service → more Female
   (Note: This reflects historical bias, not recommendations)

9. **Country Distribution**:
   - ~90% United-States
   - ~3% Mexico
   - ~2% Philippines
   - ~1% Germany, Canada, India
   - Rest: scattered across 40+ countries

10. **Race Distribution**:
   - ~85% White
   - ~10% Black
   - ~3% Asian-Pac-Islander
   - ~1% Amer-Indian-Eskimo
   - ~1% Other
"""

# Education mapping
EDUCATION_MAPPING = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8,
    "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
}

EDUCATION_NUM_TO_NAME = {v: k for k, v in EDUCATION_MAPPING.items()}

# Field groups for decomposition
FIELD_GROUPS = {
    "demographics": ["age", "sex", "race", "native.country"],
    "education": ["education", "education.num"],
    "work": ["workclass", "occupation", "hours.per.week"],
    "family": ["marital.status", "relationship"],
    "financial": ["capital.gain", "capital.loss", "fnlwgt"],
    "outcome": ["income"]
}


# ============================================================================
#                      Generation Conditions (Adult Census)
# ============================================================================

@dataclass
class GenerationCondition:
    """生成条件（修改为Adult Census）"""
    
    # 条件范围
    income: Optional[str] = None          # <=50K or >50K
    age_range: Optional[str] = None       # young, middle, senior
    education_level: Optional[str] = None # low, medium, high
    work_type: Optional[str] = None       # full-time, part-time
    
    # 目标分布
    target_distribution: Optional[Dict] = None
    
    def to_prompt(self) -> str:
        """将条件转换为提示词"""
        parts = []
        
        if self.income:
            parts.append(f"Income: {self.income}")
        if self.age_range:
            parts.append(f"Age range: {self.age_range}")
        if self.education_level:
            parts.append(f"Education level: {self.education_level}")
        if self.work_type:
            parts.append(f"Work type: {self.work_type}")
        
        return "; ".join(parts) if parts else "No special constraints"
