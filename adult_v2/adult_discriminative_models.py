"""
Adult Census Data - Discriminative Models
判别式辅助模型 - 用于增强生成数据质量
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


class AdultDiscriminativeModels:
    """
    判别式模型集合
    
    包含多个预测模型：
    1. income_predictor - 预测收入类别
    2. occupation_predictor - 预测职业
    3. hours_predictor - 预测工作时长
    4. marital_predictor - 预测婚姻状况
    5. workclass_predictor - 预测工作类型
    """
    
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.feature_columns = {}
        self.is_trained = False
    
    def train_all_models(self, data_file: str, test_size: float = 0.2, 
                        random_state: int = 42, verbose: bool = True):
        """
        训练所有判别式模型
        
        Args:
            data_file: 真实数据CSV文件路径
            test_size: 测试集比例
            random_state: 随机种子
            verbose: 是否输出训练信息
        """
        if verbose:
            print("\n" + "=" * 80)
            print("训练判别式辅助模型")
            print("=" * 80)
        
        # 加载数据
        df = pd.read_csv(data_file)
        df = df.replace('?', np.nan)
        
        if verbose:
            print(f"\n加载数据: {len(df)} 条")
        
        # 1. 训练income预测模型
        self._train_income_predictor(df, test_size, random_state, verbose)
        
        # 2. 训练occupation预测模型
        self._train_occupation_predictor(df, test_size, random_state, verbose)
        
        # 3. 训练hours预测模型
        self._train_hours_predictor(df, test_size, random_state, verbose)
        
        # 4. 训练marital预测模型
        self._train_marital_predictor(df, test_size, random_state, verbose)
        
        # 5. 训练workclass预测模型
        self._train_workclass_predictor(df, test_size, random_state, verbose)
        
        self.is_trained = True
        
        if verbose:
            print("\n" + "=" * 80)
            print("所有模型训练完成")
            print("=" * 80)
    
    def _train_income_predictor(self, df: pd.DataFrame, test_size: float,
                                random_state: int, verbose: bool):
        """训练income预测模型（最重要）"""
        if verbose:
            print("\n[1/5] 训练income预测模型...")
        
        # 特征选择
        feature_cols = ['age', 'education.num', 'hours.per.week', 'capital.gain', 
                       'capital.loss', 'sex', 'marital.status', 'occupation']
        target_col = 'income'
        
        # 准备数据
        df_clean = df[feature_cols + [target_col]].dropna()
        
        # 编码分类变量
        X = df_clean[feature_cols].copy()
        for col in ['sex', 'marital.status', 'occupation']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].dropna())
            X[col] = self.label_encoders[col].transform(X[col])
        
        # 编码目标变量
        if target_col not in self.label_encoders:
            self.label_encoders[target_col] = LabelEncoder()
            self.label_encoders[target_col].fit(df[target_col].dropna())
        y = self.label_encoders[target_col].transform(df_clean[target_col])
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 训练模型
        model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                      random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        if verbose:
            print(f"  特征: {feature_cols}")
            print(f"  训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
            print(f"  准确率: {accuracy:.3f}, F1: {f1:.3f}")
        
        self.models['income'] = model
        self.feature_columns['income'] = feature_cols
    
    def _train_occupation_predictor(self, df: pd.DataFrame, test_size: float,
                                   random_state: int, verbose: bool):
        """训练occupation预测模型"""
        if verbose:
            print("\n[2/5] 训练occupation预测模型...")
        
        # 修正：occupation应该由age、education、sex预测，不应该用income
        # 因果链：age,sex → education → occupation → income
        feature_cols = ['age', 'education.num', 'sex']
        target_col = 'occupation'
        
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = df_clean[feature_cols].copy()
        for col in ['sex']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].dropna())
            X[col] = self.label_encoders[col].transform(X[col])
        
        if target_col not in self.label_encoders:
            self.label_encoders[target_col] = LabelEncoder()
            self.label_encoders[target_col].fit(df[target_col].dropna())
        y = self.label_encoders[target_col].transform(df_clean[target_col])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = RandomForestClassifier(n_estimators=100, max_depth=15,
                                      random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if verbose:
            print(f"  特征: {feature_cols}")
            print(f"  准确率: {accuracy:.3f}")
        
        self.models['occupation'] = model
        self.feature_columns['occupation'] = feature_cols
    
    def _train_hours_predictor(self, df: pd.DataFrame, test_size: float,
                              random_state: int, verbose: bool):
        """训练hours.per.week预测模型（回归）"""
        if verbose:
            print("\n[3/5] 训练hours.per.week预测模型...")
        
        feature_cols = ['age', 'education.num', 'occupation', 'income', 'sex']
        target_col = 'hours.per.week'
        
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = df_clean[feature_cols].copy()
        for col in ['occupation', 'income', 'sex']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].dropna())
            X[col] = self.label_encoders[col].transform(X[col])
        
        y = df_clean[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                     random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        if verbose:
            print(f"  特征: {feature_cols}")
            print(f"  RMSE: {rmse:.2f} 小时")
        
        self.models['hours'] = model
        self.feature_columns['hours'] = feature_cols
    
    def _train_marital_predictor(self, df: pd.DataFrame, test_size: float,
                                random_state: int, verbose: bool):
        """训练marital.status预测模型"""
        if verbose:
            print("\n[4/5] 训练marital.status预测模型...")
        
        feature_cols = ['age', 'sex', 'education.num']
        target_col = 'marital.status'
        
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = df_clean[feature_cols].copy()
        for col in ['sex']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].dropna())
            X[col] = self.label_encoders[col].transform(X[col])
        
        if target_col not in self.label_encoders:
            self.label_encoders[target_col] = LabelEncoder()
            self.label_encoders[target_col].fit(df[target_col].dropna())
        y = self.label_encoders[target_col].transform(df_clean[target_col])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                      random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if verbose:
            print(f"  特征: {feature_cols}")
            print(f"  准确率: {accuracy:.3f}")
        
        self.models['marital'] = model
        self.feature_columns['marital'] = feature_cols
    
    def _train_workclass_predictor(self, df: pd.DataFrame, test_size: float,
                                  random_state: int, verbose: bool):
        """训练workclass预测模型"""
        if verbose:
            print("\n[5/5] 训练workclass预测模型...")
        
        # 修正：workclass和occupation是同级的，都由age、education预测
        # 因果链：age,sex → education → {workclass, occupation}
        feature_cols = ['age', 'education.num', 'sex']
        target_col = 'workclass'
        
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = df_clean[feature_cols].copy()
        for col in ['sex']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].dropna())
            X[col] = self.label_encoders[col].transform(X[col])
        
        if target_col not in self.label_encoders:
            self.label_encoders[target_col] = LabelEncoder()
            self.label_encoders[target_col].fit(df[target_col].dropna())
        y = self.label_encoders[target_col].transform(df_clean[target_col])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                      random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if verbose:
            print(f"  特征: {feature_cols}")
            print(f"  准确率: {accuracy:.3f}")
        
        self.models['workclass'] = model
        self.feature_columns['workclass'] = feature_cols
    
    def predict(self, sample: Dict, field: str) -> Optional[str]:
        """
        使用判别式模型预测字段值
        
        Args:
            sample: 样本字典
            field: 要预测的字段名
        
        Returns:
            预测值（字符串），如果无法预测返回None
        """
        if not self.is_trained or field not in self.models:
            return None
        
        try:
            # 准备特征
            feature_cols = self.feature_columns[field]
            X = []
            
            for col in feature_cols:
                value = sample.get(col)
                if value is None:
                    return None  # 缺少必要特征
                
                # 编码分类变量
                if col in self.label_encoders:
                    try:
                        value = self.label_encoders[col].transform([value])[0]
                    except:
                        return None  # 未见过的类别
                
                X.append(value)
            
            X = np.array(X).reshape(1, -1)
            
            # 预测
            model = self.models[field]
            if field == 'hours':
                # 回归模型
                pred = model.predict(X)[0]
                return int(np.clip(pred, 1, 99))
            else:
                # 分类模型
                pred = model.predict(X)[0]
                
                # 解码
                target_col = {
                    'income': 'income',
                    'occupation': 'occupation',
                    'marital': 'marital.status',
                    'workclass': 'workclass'
                }[field]
                
                return self.label_encoders[target_col].inverse_transform([pred])[0]
        
        except Exception as e:
            return None
    
    def save_models(self, model_dir: str):
        """保存所有模型"""
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型
        with open(os.path.join(model_dir, 'models.pkl'), 'wb') as f:
            pickle.dump(self.models, f)
        
        # 保存编码器
        with open(os.path.join(model_dir, 'encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # 保存特征列
        with open(os.path.join(model_dir, 'features.pkl'), 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"\n[保存] 模型已保存到: {model_dir}")
    
    def load_models(self, model_dir: str):
        """加载所有模型"""
        # 加载模型
        with open(os.path.join(model_dir, 'models.pkl'), 'rb') as f:
            self.models = pickle.load(f)
        
        # 加载编码器
        with open(os.path.join(model_dir, 'encoders.pkl'), 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        # 加载特征列
        with open(os.path.join(model_dir, 'features.pkl'), 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        self.is_trained = True
        print(f"\n[加载] 模型已加载: {len(self.models)} 个")


# ============================================================================
#                      训练脚本
# ============================================================================

def train_and_save_models(data_file: str = "archive/adult.csv",
                         model_dir: str = "adult_v2/trained_models"):
    """训练并保存所有判别式模型"""
    print("开始训练判别式辅助模型...")
    
    models = AdultDiscriminativeModels()
    models.train_all_models(data_file, test_size=0.2, verbose=True)
    models.save_models(model_dir)
    
    print("\n训练完成！")
    return models


if __name__ == "__main__":
    # 训练模型
    train_and_save_models()
