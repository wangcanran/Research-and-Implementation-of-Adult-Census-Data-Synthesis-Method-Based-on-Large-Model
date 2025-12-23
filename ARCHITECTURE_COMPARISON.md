# Adult Data Generator 架构对比

## 当前实现 vs 论文架构

### 当前 adult_data_generator.py（部分实现）

```
✓ GenerationCondition
✓ SampleWiseDecomposer (部分)
  - ✓ 条件分布学习 (9种)
  - ✓ 分步生成
  - ✓ 引导式Prompt
✓ RuleBasedVerifier (简化版Filter)
✗ DemonstrationManager (缺失)
✗ DatasetWiseScheduler (缺失)
✗ SelfInstructGenerator (缺失)
✗ Curation阶段 (完全缺失)
✗ Evaluation阶段 (完全缺失)
```

### 论文架构 data_generator.py（完整实现）

```
I. Generation
  ✓ Task Specification
  ✓ GenerationCondition
  ✓ DemonstrationManager (启发式选择)
  ✓ SampleWiseDecomposer
  ✓ DatasetWiseScheduler (目标分布调度)
  ✓ SelfInstructGenerator (迭代生成)

II. Curation
  ✓ SampleFilter
  ✓ SampleReweighter (SunGen)
  ✓ LabelEnhancer
  ✓ AuxiliaryModelEnhancer

III. Evaluation
  ✓ DirectEvaluator
  ✓ BenchmarkEvaluator
  ✓ IndirectEvaluator
```

## 重构方案

### 方案1：完全重构（推荐）

创建 `adult_data_generator_v2.py`，完全按照论文架构：

1. **复用 data_generator.py 的框架**
2. **适配 Adult 数据集的特点**
   - Task Specification: Adult Census 字段
   - GenerationCondition: age_range, education_level, income_class
   - 条件分布: P(income|education), P(occupation|education), 等
3. **保留已实现的核心改进**
   - 9种条件分布学习
   - 引导式Prompt生成
   - 条件依赖验证

### 方案2：渐进迁移

1. 保留当前 adult_data_generator.py
2. 逐步添加缺失组件：
   - 添加 DemonstrationManager
   - 添加 DatasetWiseScheduler
   - 添加 Curation 阶段
   - 添加 Evaluation 阶段

## 关键差异点

### 1. Demonstration Selection（当前缺失）

**data_generator.py:**
```python
# 启发式选择高质量示例
demos = demo_manager.select_demonstrations(condition, k=2)
# 计算质量分数：完整性 + 一致性 + 逻辑合规性
# 计算相似度：与条件的匹配程度
```

**应该添加到 Adult:**
```python
# 从真实Adult数据中选择高质量示例
# 质量评分：教育-收入一致性、婚姻-关系一致性
# 相似度：年龄接近、教育水平匹配
```

### 2. Dataset-Wise Scheduling（当前缺失）

**data_generator.py:**
```python
# 根据目标分布调度生成条件
condition = scheduler.get_next_condition()
# 确保生成的数据分布匹配目标
```

**应该添加到 Adult:**
```python
target_distribution = {
    "age_range": {"young": 0.3, "middle": 0.5, "senior": 0.2},
    "education": {"low": 0.3, "medium": 0.5, "high": 0.2},
    "income": {"<=50K": 0.76, ">50K": 0.24}
}
```

### 3. Curation阶段（完全缺失）

**data_generator.py 有完整的策展流程:**
```python
# Filter: 过滤低质量样本
# Reweight: SunGen双循环重加权
# Label Enhance: 标签平滑/增强
# Auxiliary Model: 判别式模型增强
```

### 4. Evaluation阶段（完全缺失）

**data_generator.py 有三层评估:**
```python
# Direct: 格式正确性、逻辑一致性、事实性
# Benchmark: 与真实数据分布对比
# Indirect: 下游任务性能
```

## 建议行动

**立即行动：**
创建按照论文架构的 `adult_data_generator_v2.py`

**优势：**
1. 完全符合论文方法
2. 可复现论文结果
3. 支持完整的质量保证流程
4. 可进行科学评估

**保留当前工作的价值：**
- 9种条件分布学习 ✓
- 引导式Prompt策略 ✓
- 教育映射和业务规则 ✓
