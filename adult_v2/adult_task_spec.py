"""
Adult Census Data - Task Specification & Generation Conditions
任务规范和生成条件
"""

from dataclasses import dataclass
from typing import Optional, Dict, List

# ============================================================================
#                      Task Specification
# ============================================================================

# Education Mapping (教育程度映射)
EDUCATION_MAPPING = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8,
    "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
}

EDUCATION_NUM_TO_NAME = {v: k for k, v in EDUCATION_MAPPING.items()}

# Field Groups (字段分组 - 按依赖关系排序)
FIELD_GROUPS = {
    "demographics": ["age", "sex", "race", "native.country"],
    "education": ["education", "education.num"],
    "work": ["workclass", "occupation", "hours.per.week"],
    "family": ["marital.status", "relationship"],
    "financial": ["capital.gain", "capital.loss", "fnlwgt"],
    "outcome": ["income"]
}

# All Fields
ALL_FIELDS = [
    "age", "workclass", "fnlwgt", "education", "education.num",
    "marital.status", "occupation", "relationship", "race", "sex",
    "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"
]

# Categorical Field Values
CATEGORICAL_VALUES = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                  "Local-gov", "State-gov", "Without-pay", "Never-worked", "?"],
    "education": list(EDUCATION_MAPPING.keys()),
    "marital.status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
                       "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales",
                   "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                   "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                   "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces", "?"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family",
                     "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Male", "Female"],
    "native.country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
                       "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan",
                       "Greece", "South", "China", "Cuba", "Iran", "Honduras",
                       "Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
                       "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                       "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary",
                       "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
                       "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"],
    "income": ["<=50K", ">50K"]
}

# Numerical Field Ranges
NUMERICAL_RANGES = {
    "age": (17, 90),
    "education.num": (1, 16),
    "hours.per.week": (1, 99),
    "capital.gain": (0, 99999),
    "capital.loss": (0, 4356),
    "fnlwgt": (12285, 1484705)
}


# ============================================================================
#                      Generation Conditions
# ============================================================================

@dataclass
class GenerationCondition:
    """生成条件 - Conditioning Scope & Values"""
    
    # Age Conditioning
    age_range: Optional[str] = None  # "young" (17-30), "middle" (31-55), "senior" (56-90)
    
    # Education Conditioning
    education_level: Optional[str] = None  # "low" (<=8), "medium" (9-12), "high" (>=13)
    
    # Gender Conditioning
    gender: Optional[str] = None  # "Male" or "Female"
    
    # Income Conditioning
    income_class: Optional[str] = None  # "<=50K" or ">50K"
    
    # Work Conditioning
    occupation_type: Optional[str] = None  # "professional", "service", "manual"
    
    # Marital Status Conditioning
    marital_status: Optional[str] = None  # "married", "single", "divorced"
    
    def __repr__(self):
        parts = []
        if self.age_range:
            parts.append(f"age={self.age_range}")
        if self.education_level:
            parts.append(f"edu={self.education_level}")
        if self.gender:
            parts.append(f"sex={self.gender}")
        if self.income_class:
            parts.append(f"income={self.income_class}")
        if self.occupation_type:
            parts.append(f"occ={self.occupation_type}")
        if self.marital_status:
            parts.append(f"marital={self.marital_status}")
        
        return f"Condition({', '.join(parts) if parts else 'unconditioned'})"


# ============================================================================
#                      Business Rules & Validation
# ============================================================================

def validate_education_mapping(education: str, education_num: int) -> bool:
    """验证教育程度和教育年限的映射关系"""
    return EDUCATION_MAPPING.get(education) == education_num


def validate_marital_relationship(marital: str, sex: str, relationship: str) -> bool:
    """验证婚姻-性别-关系的一致性"""
    if marital == "Married-civ-spouse":
        if sex == "Male" and relationship not in ["Husband"]:
            return False
        if sex == "Female" and relationship not in ["Wife"]:
            return False
    elif marital == "Never-married":
        if relationship not in ["Not-in-family", "Own-child", "Other-relative", "Unmarried"]:
            return False
    elif marital in ["Divorced", "Separated", "Widowed"]:
        if relationship not in ["Not-in-family", "Unmarried", "Other-relative"]:
            return False
    
    return True


def validate_numerical_range(field: str, value) -> bool:
    """验证数值字段的范围"""
    if field not in NUMERICAL_RANGES:
        return True
    
    try:
        val = int(value) if isinstance(value, str) else value
        min_val, max_val = NUMERICAL_RANGES[field]
        return min_val <= val <= max_val
    except:
        return False


def validate_sample(sample: Dict) -> tuple[bool, List[str]]:
    """
    验证样本的完整性和一致性
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # 1. Field Completeness
    for field in ALL_FIELDS:
        if field not in sample or sample[field] is None or sample[field] == "":
            errors.append(f"Missing field: {field}")
    
    if errors:  # 如果有缺失字段，后续检查会出错
        return False, errors
    
    # 2. Education Mapping
    education = sample.get("education", "")
    education_num = sample.get("education.num")
    if not validate_education_mapping(education, education_num):
        expected_num = EDUCATION_MAPPING.get(education, "?")
        errors.append(f"Education mapping error: {education} should be {expected_num}, got {education_num}")
    
    # 3. Marital-Relationship-Sex Consistency
    marital = sample.get("marital.status", "")
    sex = sample.get("sex", "")
    relationship = sample.get("relationship", "")
    if not validate_marital_relationship(marital, sex, relationship):
        errors.append(f"Marital-Sex-Relationship inconsistency: {marital} + {sex} -> {relationship}")
    
    # 4. Numerical Ranges
    for field in NUMERICAL_RANGES.keys():
        if field in sample:
            if not validate_numerical_range(field, sample[field]):
                min_val, max_val = NUMERICAL_RANGES[field]
                errors.append(f"{field} out of range: {sample[field]} (expected {min_val}-{max_val})")
    
    # 5. Categorical Values
    for field, valid_values in CATEGORICAL_VALUES.items():
        if field in sample and sample[field] not in valid_values:
            errors.append(f"{field} invalid value: {sample[field]} (not in valid set)")
    
    return len(errors) == 0, errors
