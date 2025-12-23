"""
Adult Census Data - Discriminative Models with Multiple ML Algorithms
判别式辅助模型 - 支持多种机器学习算法的训练和比较
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# 尝试导入XGBoost和LightGBM（可选）
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost未安装，将跳过XGBoost模型")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠ LightGBM未安装，将跳过LightGBM模型")


class MLModelTrainer:
    """
    机器学习模型训练器
    支持多种算法并自动选择最佳模型
    """
    
    @staticmethod
    def get_classification_models():
        """获取所有可用的分类模型"""
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0
            )
        
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=-1
            )
        
        return models
    
    @staticmethod
    def get_regression_models():
        """获取所有可用的回归模型"""
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0
            )
        
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=-1
            )
        
        return models
    
    @staticmethod
    def train_and_select_best(X_train, y_train, X_test, y_test, 
                             task_type='classification', verbose=True):
        """
        训练多个模型并选择最佳模型
        
        Args:
            task_type: 'classification' or 'regression'
            
        Returns:
            best_model, best_score, all_results
        """
        if task_type == 'classification':
            models = MLModelTrainer.get_classification_models()
        else:
            models = MLModelTrainer.get_regression_models()
        
        results = {}
        best_model = None
        best_score = -float('inf')
        best_name = None
        
        if verbose:
            print(f"    比较 {len(models)} 个模型...")
        
        for name, model in models.items():
            try:
                # 训练模型
                model.fit(X_train, y_train)
                
                # 评估
                if task_type == 'classification':
                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    results[name] = {'accuracy': score, 'f1': f1}
                else:
                    y_pred = model.predict(X_test)
                    score = -mean_squared_error(y_test, y_pred)  # 负MSE，越大越好
                    rmse = np.sqrt(-score)
                    r2 = r2_score(y_test, y_pred)
                    results[name] = {'rmse': rmse, 'r2': r2}
                
                if verbose:
                    if task_type == 'classification':
                        print(f"      {name:20s} - Acc: {results[name]['accuracy']:.3f}, F1: {results[name]['f1']:.3f}")
                    else:
                        print(f"      {name:20s} - RMSE: {results[name]['rmse']:.2f}, R²: {results[name]['r2']:.3f}")
                
                # 更新最佳模型
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                if verbose:
                    print(f"      {name:20s} - 训练失败: {e}")
        
        if verbose and best_name:
            print(f"    ✓ 最佳模型: {best_name}")
        
        return best_model, best_name, results


class AdultDiscriminativeModelsML:
    """
    判别式模型集合（多种ML算法）
    自动选择每个任务的最佳算法
    """
    
    def __init__(self):
        self.models = {}
        self.model_names = {}  # 存储每个任务选择的算法名称
        self.label_encoders = {}
        self.scalers = {}  # 特征标准化器
        self.feature_columns = {}
        self.is_trained = False
    
    def train_all_models(self, data_file: str, test_size: float = 0.2,
                        random_state: int = 42, use_scaling: bool = True,
                        verbose: bool = True):
        """
        训练所有判别式模型，每个任务自动选择最佳算法
        
        Args:
            data_file: 真实数据CSV文件路径
            test_size: 测试集比例
            random_state: 随机种子
            use_scaling: 是否对数值特征进行标准化
            verbose: 是否输出训练信息
        """
        if verbose:
            print("\n" + "=" * 80)
            print("训练判别式辅助模型（多种ML算法比较）")
            print("=" * 80)
        
        # 加载数据
        df = pd.read_csv(data_file)
        df = df.replace('?', np.nan)
        
        if verbose:
            print(f"\n加载数据: {len(df)} 条")
        
        self.use_scaling = use_scaling
        
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
            print("\n选择的最佳模型：")
            for task, name in self.model_names.items():
                print(f"  {task:15s}: {name}")
    
    def _prepare_features(self, df: pd.DataFrame, feature_cols: List[str],
                         task_name: str, is_training: bool = True):
        """准备特征（编码+标准化）"""
        X = df[feature_cols].copy()
        
        # 编码分类变量
        for col in feature_cols:
            if df[col].dtype == 'object':
                if is_training:
                    encoder_key = f"{task_name}_{col}"
                    if encoder_key not in self.label_encoders:
                        self.label_encoders[encoder_key] = LabelEncoder()
                        self.label_encoders[encoder_key].fit(df[col].dropna())
                    X[col] = self.label_encoders[encoder_key].transform(X[col])
                else:
                    encoder_key = f"{task_name}_{col}"
                    X[col] = self.label_encoders[encoder_key].transform(X[col])
        
        # 标准化数值特征
        if self.use_scaling:
            if is_training:
                self.scalers[task_name] = StandardScaler()
                X = pd.DataFrame(
                    self.scalers[task_name].fit_transform(X),
                    columns=X.columns
                )
            else:
                X = pd.DataFrame(
                    self.scalers[task_name].transform(X),
                    columns=X.columns
                )
        
        return X
    
    def _train_income_predictor(self, df: pd.DataFrame, test_size: float,
                                random_state: int, verbose: bool):
        """训练income预测模型"""
        if verbose:
            print("\n[1/5] 训练income预测模型（核心任务）...")
        
        feature_cols = ['age', 'education.num', 'hours.per.week', 'capital.gain',
                       'capital.loss', 'sex', 'marital.status', 'occupation']
        target_col = 'income'
        
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = self._prepare_features(df_clean, feature_cols, 'income', is_training=True)
        
        if target_col not in self.label_encoders:
            self.label_encoders[target_col] = LabelEncoder()
            self.label_encoders[target_col].fit(df[target_col].dropna())
        y = self.label_encoders[target_col].transform(df_clean[target_col])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        best_model, best_name, results = MLModelTrainer.train_and_select_best(
            X_train.values, y_train, X_test.values, y_test,
            task_type='classification', verbose=verbose
        )
        
        self.models['income'] = best_model
        self.model_names['income'] = best_name
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
        
        X = self._prepare_features(df_clean, feature_cols, 'occupation', is_training=True)
        
        if target_col not in self.label_encoders:
            self.label_encoders[target_col] = LabelEncoder()
            self.label_encoders[target_col].fit(df[target_col].dropna())
        y = self.label_encoders[target_col].transform(df_clean[target_col])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        best_model, best_name, results = MLModelTrainer.train_and_select_best(
            X_train.values, y_train, X_test.values, y_test,
            task_type='classification', verbose=verbose
        )
        
        self.models['occupation'] = best_model
        self.model_names['occupation'] = best_name
        self.feature_columns['occupation'] = feature_cols
    
    def _train_hours_predictor(self, df: pd.DataFrame, test_size: float,
                              random_state: int, verbose: bool):
        """训练hours.per.week预测模型"""
        if verbose:
            print("\n[3/5] 训练hours.per.week预测模型...")
        
        feature_cols = ['age', 'education.num', 'occupation', 'income', 'sex']
        target_col = 'hours.per.week'
        
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = self._prepare_features(df_clean, feature_cols, 'hours', is_training=True)
        y = df_clean[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        best_model, best_name, results = MLModelTrainer.train_and_select_best(
            X_train.values, y_train, X_test.values, y_test,
            task_type='regression', verbose=verbose
        )
        
        self.models['hours'] = best_model
        self.model_names['hours'] = best_name
        self.feature_columns['hours'] = feature_cols
    
    def _train_marital_predictor(self, df: pd.DataFrame, test_size: float,
                                random_state: int, verbose: bool):
        """训练marital.status预测模型"""
        if verbose:
            print("\n[4/5] 训练marital.status预测模型...")
        
        feature_cols = ['age', 'sex', 'education.num']
        target_col = 'marital.status'
        
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = self._prepare_features(df_clean, feature_cols, 'marital', is_training=True)
        
        if target_col not in self.label_encoders:
            self.label_encoders[target_col] = LabelEncoder()
            self.label_encoders[target_col].fit(df[target_col].dropna())
        y = self.label_encoders[target_col].transform(df_clean[target_col])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        best_model, best_name, results = MLModelTrainer.train_and_select_best(
            X_train.values, y_train, X_test.values, y_test,
            task_type='classification', verbose=verbose
        )
        
        self.models['marital'] = best_model
        self.model_names['marital'] = best_name
        self.feature_columns['marital'] = feature_cols
    
    def _train_workclass_predictor(self, df: pd.DataFrame, test_size: float,
                                  random_state: int, verbose: bool):
        """训练workclass预测模型"""
        if verbose:
            print("\n[5/5] 训练workclass预测模型...")
        
        # 修正：workclass和occupation是同级的，都由age、education预测
        feature_cols = ['age', 'education.num', 'sex']
        target_col = 'workclass'
        
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = self._prepare_features(df_clean, feature_cols, 'workclass', is_training=True)
        
        if target_col not in self.label_encoders:
            self.label_encoders[target_col] = LabelEncoder()
            self.label_encoders[target_col].fit(df[target_col].dropna())
        y = self.label_encoders[target_col].transform(df_clean[target_col])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        best_model, best_name, results = MLModelTrainer.train_and_select_best(
            X_train.values, y_train, X_test.values, y_test,
            task_type='classification', verbose=verbose
        )
        
        self.models['workclass'] = best_model
        self.model_names['workclass'] = best_name
        self.feature_columns['workclass'] = feature_cols
    
    def predict(self, sample: Dict, field: str) -> Optional[str]:
        """使用判别式模型预测字段值（与原版接口兼容）"""
        if not self.is_trained or field not in self.models:
            return None
        
        try:
            feature_cols = self.feature_columns[field]
            X_dict = {}
            
            for col in feature_cols:
                value = sample.get(col)
                if value is None:
                    return None
                X_dict[col] = value
            
            # 创建DataFrame以便使用_prepare_features
            X_df = pd.DataFrame([X_dict])
            
            # 准备特征（需要临时使用已训练的编码器）
            for col in feature_cols:
                if X_df[col].dtype == 'object':
                    encoder_key = f"{field}_{col}"
                    try:
                        X_df[col] = self.label_encoders[encoder_key].transform(X_df[col])
                    except:
                        return None
            
            # 标准化
            if self.use_scaling and field in self.scalers:
                X = self.scalers[field].transform(X_df.values)
            else:
                X = X_df.values
            
            # 预测
            model = self.models[field]
            if field == 'hours':
                pred = model.predict(X)[0]
                return int(np.clip(pred, 1, 99))
            else:
                pred = model.predict(X)[0]
                
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
        
        with open(os.path.join(model_dir, 'models_ml.pkl'), 'wb') as f:
            pickle.dump(self.models, f)
        
        with open(os.path.join(model_dir, 'model_names.pkl'), 'wb') as f:
            pickle.dump(self.model_names, f)
        
        with open(os.path.join(model_dir, 'encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open(os.path.join(model_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open(os.path.join(model_dir, 'features.pkl'), 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        with open(os.path.join(model_dir, 'use_scaling.pkl'), 'wb') as f:
            pickle.dump(self.use_scaling, f)
        
        print(f"\n[保存] 模型已保存到: {model_dir}")
    
    def load_models(self, model_dir: str):
        """加载所有模型"""
        with open(os.path.join(model_dir, 'models_ml.pkl'), 'rb') as f:
            self.models = pickle.load(f)
        
        with open(os.path.join(model_dir, 'model_names.pkl'), 'rb') as f:
            self.model_names = pickle.load(f)
        
        with open(os.path.join(model_dir, 'encoders.pkl'), 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(os.path.join(model_dir, 'scalers.pkl'), 'rb') as f:
            self.scalers = pickle.load(f)
        
        with open(os.path.join(model_dir, 'features.pkl'), 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        with open(os.path.join(model_dir, 'use_scaling.pkl'), 'rb') as f:
            self.use_scaling = pickle.load(f)
        
        self.is_trained = True
        print(f"\n[加载] 模型已加载: {len(self.models)} 个")
        print("使用的算法:")
        for task, name in self.model_names.items():
            print(f"  {task:15s}: {name}")


def train_and_save_models_ml(data_file: str = "archive/adult.csv",
                             model_dir: str = "adult_v2/trained_models_ml",
                             use_scaling: bool = True):
    """训练并保存所有判别式模型（多种ML算法）"""
    print("开始训练判别式辅助模型（多种ML算法比较）...")
    
    models = AdultDiscriminativeModelsML()
    models.train_all_models(data_file, test_size=0.2, use_scaling=use_scaling, verbose=True)
    models.save_models(model_dir)
    
    print("\n训练完成！")
    return models


if __name__ == "__main__":
    train_and_save_models_ml()
