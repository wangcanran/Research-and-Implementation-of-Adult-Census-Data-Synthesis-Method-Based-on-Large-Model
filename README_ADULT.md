# Adult Census 合成数据生成器

基于大语言模型(LLM)的Adult Census收入数据集合成数据生成器，采用与原始`data_generator.py`相同的架构和策略。

## 📋 目录结构

```
├── archive/
│   └── adult.csv                    # 原始Adult Census数据集
├── adult_config.py                  # 配置文件（API密钥等）
├── adult_data_generator.py          # 主生成器代码
├── generate_adult_samples.py        # 示例脚本
└── README_ADULT.md                  # 本文档
```

## 🎯 核心特性

### 1. 基于LLM的生成策略
- **Task Specification**: 详细的数据表结构和业务规则说明
- **Generation Conditions**: 支持多维度条件约束（年龄、教育、收入、性别等）
- **In-Context Demonstrations**: 智能示例选择机制
- **Sample-Wise Decomposition**: 分组生成确保字段间逻辑一致性

### 2. 启发式示例选择
- **质量评分**: 基于字段完整性、逻辑一致性的多维度评分
- **相似度计算**: 选择与生成条件最匹配的示例
- **不确定性评估**: 优先选择难度适中的样本
- **多样性保证**: 避免选择过于相似的示例

### 3. 规则验证
- 年龄范围验证 (17-90岁)
- 教育程度与年限一致性
- 婚姻状况-家庭关系-性别三元一致性
- 工作时长合理性检查
- 资本收益/损失范围验证

### 4. 统计学习
从真实数据中学习：
- 年龄分布特征
- 工作时长分布
- 收入类别比例
- 自动适应数据分布

## 🚀 快速开始

### 1. 配置API密钥

编辑 `adult_config.py`:

```python
OPENAI_API_KEY = "your-actual-api-key"
OPENAI_API_BASE = "https://api.openai.com/v1"
FIXED_MODEL_NAME = "gpt-4o-mini"  # 或其他模型
```

### 2. 运行示例脚本

```bash
python generate_adult_samples.py
```

该脚本将生成4个场景的数据：
- 高收入高学历中年人群
- 低收入年轻女性
- 老年人群
- 完全随机样本

### 3. 自定义生成

```python
from adult_data_generator import AdultDataGenerator, GenerationCondition

# 初始化生成器
generator = AdultDataGenerator(
    sample_file="archive/adult.csv",
    use_heuristic=True,
    verbose=True
)

# 定义生成条件
condition = GenerationCondition(
    age_range="middle",         # young/middle/senior
    education_level="high",     # low/medium/high
    income_class=">50K",        # <=50K/>50K
    gender="Male",              # Male/Female
    marital_status=None         # 或具体值
)

# 生成样本
samples = generator.generate_batch(100, condition)

# 保存到CSV
generator.save_to_csv(samples, "my_synthetic_data.csv")
```

## 📊 数据字段说明

| 字段 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| age | int | 年龄 | 38 |
| workclass | str | 工作类型 | Private, Self-emp-inc |
| fnlwgt | int | 最终权重 | 215646 |
| education | str | 教育程度 | Bachelors, Masters |
| education.num | int | 教育年限 | 13 (对应Bachelors) |
| marital.status | str | 婚姻状况 | Married-civ-spouse |
| occupation | str | 职业 | Exec-managerial |
| relationship | str | 家庭关系 | Husband, Wife |
| race | str | 种族 | White, Black, Asian-Pac-Islander |
| sex | str | 性别 | Male, Female |
| capital.gain | int | 资本收益 | 0-99999 |
| capital.loss | int | 资本损失 | 0-4356 |
| hours.per.week | int | 每周工时 | 40 |
| native.country | str | 原籍国家 | United-States |
| income | str | 收入类别 | <=50K, >50K |

## 🔧 高级配置

### GenerationCondition 参数

```python
GenerationCondition(
    age_range="middle",        # 年龄范围
                               # - "young": 17-30岁
                               # - "middle": 31-55岁  
                               # - "senior": 56-90岁
    
    education_level="high",    # 教育水平
                               # - "low": <=12年
                               # - "medium": 9-12年
                               # - "high": >=13年
    
    income_class=">50K",       # 收入类别
                               # - "<=50K": 低收入
                               # - ">50K": 高收入
    
    gender="Male",             # 性别约束
    marital_status=None        # 婚姻状况约束（可选）
)
```

### 生成器参数

```python
AdultDataGenerator(
    sample_file="archive/adult.csv",  # 真实数据文件路径
    use_heuristic=True,               # 是否启用启发式选择
    verbose=True                      # 是否显示详细日志
)
```

## 📈 生成质量保证

### 1. 字段组分步生成
生成顺序：demographics → education → work → family → financial → outcome

确保字段间依赖关系正确：
- 年龄 → 教育程度
- 教育程度 → 职业类型
- 性别+婚姻状况 → 家庭关系
- 综合因素 → 收入类别

### 2. 业务规则约束
- 高学历（Doctorate/Masters）→ 专业职业（Prof-specialty/Exec-managerial）
- 已婚男性 → Husband；已婚女性 → Wife
- 长工时(≥45h) + 高学历 → 高收入倾向
- 资本收益大多为0（符合真实分布）

### 3. 统计分布匹配
从真实数据学习统计特征，生成数据分布与原始数据一致

## 🎨 应用场景

1. **数据增强**: 为机器学习模型生成更多训练样本
2. **隐私保护**: 生成符合统计特征但不泄露真实个人信息的数据
3. **边缘案例测试**: 生成特定条件下的测试数据
4. **数据平衡**: 为少数类别生成更多样本
5. **研究分析**: 生成假设场景数据用于分析

## 📝 示例输出

```csv
age,workclass,fnlwgt,education,education.num,marital.status,occupation,relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country,income
45,Private,234567,Masters,14,Married-civ-spouse,Exec-managerial,Husband,White,Male,15024,0,50,United-States,>50K
22,Private,189450,Some-college,10,Never-married,Sales,Not-in-family,White,Female,0,0,35,United-States,<=50K
```

## ⚠️ 注意事项

1. **API费用**: 使用OpenAI API会产生费用，建议先小批量测试
2. **生成速度**: LLM调用需要时间，大批量生成可能较慢
3. **数据验证**: 生成后建议人工抽查验证质量
4. **统计偏差**: 虽有统计学习，但生成数据可能与真实数据有细微差异

## 🔗 相关资源

- 原始数据集: [UCI Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult)
- 基础代码架构: `data_generator.py` (门架交易数据生成器)

## 📄 许可证

本项目遵循MIT许可证

---

**作者**: 基于data_generator.py架构改编  
**版本**: 1.0.0  
**最后更新**: 2024-12
