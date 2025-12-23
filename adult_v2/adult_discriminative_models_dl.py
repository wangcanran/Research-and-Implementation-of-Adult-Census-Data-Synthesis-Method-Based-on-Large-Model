"""
Adult Census Data - Discriminative Models with Deep Learning
判别式辅助模型 - 支持神经网络（PyTorch）
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("⚠ PyTorch未安装，神经网络功能不可用")
    print("  安装方法: pip install torch")


# ============================================================================
#                      神经网络模型定义
# ============================================================================

if HAS_PYTORCH:
    
    class MLPClassifier(nn.Module):
        """多层感知机分类器"""
        
        def __init__(self, input_dim: int, num_classes: int, 
                     hidden_dims: List[int] = [128, 64, 32],
                     dropout: float = 0.3):
            super(MLPClassifier, self).__init__()
            
            layers = []
            prev_dim = input_dim
            
            # 隐藏层
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            # 输出层
            layers.append(nn.Linear(prev_dim, num_classes))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    
    class MLPRegressor(nn.Module):
        """多层感知机回归器"""
        
        def __init__(self, input_dim: int, 
                     hidden_dims: List[int] = [128, 64, 32],
                     dropout: float = 0.3):
            super(MLPRegressor, self).__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x).squeeze()


    class DeepLearningTrainer:
        """深度学习模型训练器"""
        
        @staticmethod
        def train_classifier(X_train, y_train, X_test, y_test,
                           num_classes: int,
                           epochs: int = 50,
                           batch_size: int = 128,
                           lr: float = 0.001,
                           verbose: bool = True):
            """训练分类器"""
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 准备数据
            X_train_t = torch.FloatTensor(X_train).to(device)
            y_train_t = torch.LongTensor(y_train).to(device)
            X_test_t = torch.FloatTensor(X_test).to(device)
            y_test_t = torch.LongTensor(y_test).to(device)
            
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # 创建模型
            model = MLPClassifier(
                input_dim=X_train.shape[1],
                num_classes=num_classes,
                hidden_dims=[128, 64, 32],
                dropout=0.3
            ).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            
            # 训练
            best_acc = 0.0
            best_model_state = None
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 验证
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_t)
                    test_loss = criterion(test_outputs, y_test_t)
                    _, predicted = torch.max(test_outputs, 1)
                    acc = (predicted == y_test_t).float().mean().item()
                
                scheduler.step(test_loss)
                
                # 保存最佳模型
                if acc > best_acc:
                    best_acc = acc
                    best_model_state = model.state_dict().copy()
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"        Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, "
                          f"Val Acc: {acc:.3f}")
            
            # 加载最佳模型
            model.load_state_dict(best_model_state)
            model.eval()
            
            # 最终评估
            with torch.no_grad():
                outputs = model(X_test_t)
                _, predicted = torch.max(outputs, 1)
                y_pred = predicted.cpu().numpy()
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
            
            return model.cpu(), accuracy, f1
        
        @staticmethod
        def train_regressor(X_train, y_train, X_test, y_test,
                          epochs: int = 50,
                          batch_size: int = 128,
                          lr: float = 0.001,
                          verbose: bool = True):
            """训练回归器"""
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 准备数据
            X_train_t = torch.FloatTensor(X_train).to(device)
            y_train_t = torch.FloatTensor(y_train).to(device)
            X_test_t = torch.FloatTensor(X_test).to(device)
            y_test_t = torch.FloatTensor(y_test).to(device)
            
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # 创建模型
            model = MLPRegressor(
                input_dim=X_train.shape[1],
                hidden_dims=[128, 64, 32],
                dropout=0.3
            ).to(device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            
            # 训练
            best_loss = float('inf')
            best_model_state = None
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 验证
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_t)
                    test_loss = criterion(test_outputs, y_test_t)
                
                scheduler.step(test_loss)
                
                # 保存最佳模型
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_model_state = model.state_dict().copy()
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"        Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, "
                          f"Val Loss: {test_loss:.4f}")
            
            # 加载最佳模型
            model.load_state_dict(best_model_state)
            model.eval()
            
            # 最终评估
            with torch.no_grad():
                y_pred = model(X_test_t).cpu().numpy()
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
            
            return model.cpu(), rmse, r2


class AdultDiscriminativeModelsDL:
    """
    判别式模型集合（深度学习版本）
    使用PyTorch神经网络
    """
    
    def __init__(self):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch未安装，无法使用深度学习模型")
        
        self.models = {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_columns = {}
        self.num_classes = {}
        self.is_trained = False
    
    def train_all_models(self, data_file: str, test_size: float = 0.2,
                        random_state: int = 42, epochs: int = 50,
                        verbose: bool = True):
        """
        训练所有判别式模型（使用神经网络）
        
        Args:
            data_file: 真实数据CSV文件路径
            test_size: 测试集比例
            random_state: 随机种子
            epochs: 训练轮数
            verbose: 是否输出训练信息
        """
        if verbose:
            print("\n" + "=" * 80)
            print("训练判别式辅助模型（深度学习 - PyTorch）")
            print("=" * 80)
            device = 'GPU' if torch.cuda.is_available() else 'CPU'
            print(f"使用设备: {device}")
        
        # 加载数据
        df = pd.read_csv(data_file)
        df = df.replace('?', np.nan)
        
        if verbose:
            print(f"加载数据: {len(df)} 条")
        
        # 训练各个模型
        self._train_income_predictor(df, test_size, random_state, epochs, verbose)
        self._train_occupation_predictor(df, test_size, random_state, epochs, verbose)
        self._train_hours_predictor(df, test_size, random_state, epochs, verbose)
        self._train_marital_predictor(df, test_size, random_state, epochs, verbose)
        self._train_workclass_predictor(df, test_size, random_state, epochs, verbose)
        
        self.is_trained = True
        
        if verbose:
            print("\n" + "=" * 80)
            print("所有模型训练完成（深度学习）")
            print("=" * 80)
    
    def _prepare_features(self, df: pd.DataFrame, feature_cols: List[str],
                         task_name: str, is_training: bool = True):
        """准备特征（编码+标准化）"""
        X = df[feature_cols].copy()
        
        # 编码分类变量
        for col in feature_cols:
            if df[col].dtype == 'object':
                encoder_key = f"{task_name}_{col}"
                if is_training:
                    if encoder_key not in self.label_encoders:
                        self.label_encoders[encoder_key] = LabelEncoder()
                        self.label_encoders[encoder_key].fit(df[col].dropna())
                    X[col] = self.label_encoders[encoder_key].transform(X[col])
                else:
                    X[col] = self.label_encoders[encoder_key].transform(X[col])
        
        # 标准化
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
                                random_state: int, epochs: int, verbose: bool):
        """训练income预测模型（神经网络）"""
        if verbose:
            print("\n[1/5] 训练income预测模型（MLP分类器）...")
        
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
            X.values, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        num_classes = len(np.unique(y))
        model, accuracy, f1 = DeepLearningTrainer.train_classifier(
            X_train, y_train, X_test, y_test,
            num_classes=num_classes,
            epochs=epochs,
            verbose=verbose
        )
        
        if verbose:
            print(f"      最终性能 - Acc: {accuracy:.3f}, F1: {f1:.3f}")
        
        self.models['income'] = model
        self.num_classes['income'] = num_classes
        self.feature_columns['income'] = feature_cols
    
    def _train_occupation_predictor(self, df: pd.DataFrame, test_size: float,
                                   random_state: int, epochs: int, verbose: bool):
        """训练occupation预测模型"""
        if verbose:
            print("\n[2/5] 训练occupation预测模型（MLP分类器）...")
        
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
            X.values, y, test_size=test_size, random_state=random_state
        )
        
        num_classes = len(np.unique(y))
        model, accuracy, f1 = DeepLearningTrainer.train_classifier(
            X_train, y_train, X_test, y_test,
            num_classes=num_classes,
            epochs=epochs,
            verbose=verbose
        )
        
        if verbose:
            print(f"      最终性能 - Acc: {accuracy:.3f}, F1: {f1:.3f}")
        
        self.models['occupation'] = model
        self.num_classes['occupation'] = num_classes
        self.feature_columns['occupation'] = feature_cols
    
    def _train_hours_predictor(self, df: pd.DataFrame, test_size: float,
                              random_state: int, epochs: int, verbose: bool):
        """训练hours预测模型（回归）"""
        if verbose:
            print("\n[3/5] 训练hours.per.week预测模型（MLP回归器）...")
        
        feature_cols = ['age', 'education.num', 'occupation', 'income', 'sex']
        target_col = 'hours.per.week'
        
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = self._prepare_features(df_clean, feature_cols, 'hours', is_training=True)
        y = df_clean[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=test_size, random_state=random_state
        )
        
        model, rmse, r2 = DeepLearningTrainer.train_regressor(
            X_train, y_train, X_test, y_test,
            epochs=epochs,
            verbose=verbose
        )
        
        if verbose:
            print(f"      最终性能 - RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        self.models['hours'] = model
        self.feature_columns['hours'] = feature_cols
    
    def _train_marital_predictor(self, df: pd.DataFrame, test_size: float,
                                random_state: int, epochs: int, verbose: bool):
        """训练marital预测模型"""
        if verbose:
            print("\n[4/5] 训练marital.status预测模型（MLP分类器）...")
        
        feature_cols = ['age', 'sex', 'education.num']
        target_col = 'marital.status'
        
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = self._prepare_features(df_clean, feature_cols, 'marital', is_training=True)
        
        if target_col not in self.label_encoders:
            self.label_encoders[target_col] = LabelEncoder()
            self.label_encoders[target_col].fit(df[target_col].dropna())
        y = self.label_encoders[target_col].transform(df_clean[target_col])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=test_size, random_state=random_state
        )
        
        num_classes = len(np.unique(y))
        model, accuracy, f1 = DeepLearningTrainer.train_classifier(
            X_train, y_train, X_test, y_test,
            num_classes=num_classes,
            epochs=epochs,
            verbose=verbose
        )
        
        if verbose:
            print(f"      最终性能 - Acc: {accuracy:.3f}, F1: {f1:.3f}")
        
        self.models['marital'] = model
        self.num_classes['marital'] = num_classes
        self.feature_columns['marital'] = feature_cols
    
    def _train_workclass_predictor(self, df: pd.DataFrame, test_size: float,
                                  random_state: int, epochs: int, verbose: bool):
        """训练workclass预测模型"""
        if verbose:
            print("\n[5/5] 训练workclass预测模型（MLP分类器）...")
        
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
            X.values, y, test_size=test_size, random_state=random_state
        )
        
        num_classes = len(np.unique(y))
        model, accuracy, f1 = DeepLearningTrainer.train_classifier(
            X_train, y_train, X_test, y_test,
            num_classes=num_classes,
            epochs=epochs,
            verbose=verbose
        )
        
        if verbose:
            print(f"      最终性能 - Acc: {accuracy:.3f}, F1: {f1:.3f}")
        
        self.models['workclass'] = model
        self.num_classes['workclass'] = num_classes
        self.feature_columns['workclass'] = feature_cols
    
    def predict(self, sample: Dict, field: str) -> Optional[str]:
        """使用神经网络预测字段值"""
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
            
            # 创建DataFrame
            X_df = pd.DataFrame([X_dict])
            
            # 编码分类变量
            for col in feature_cols:
                if X_df[col].dtype == 'object':
                    encoder_key = f"{field}_{col}"
                    try:
                        X_df[col] = self.label_encoders[encoder_key].transform(X_df[col])
                    except:
                        return None
            
            # 标准化
            X = self.scalers[field].transform(X_df.values)
            X_tensor = torch.FloatTensor(X)
            
            # 预测
            model = self.models[field]
            model.eval()
            with torch.no_grad():
                if field == 'hours':
                    pred = model(X_tensor).item()
                    return int(np.clip(pred, 1, 99))
                else:
                    outputs = model(X_tensor)
                    _, predicted = torch.max(outputs, 1)
                    pred = predicted.item()
                    
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
        
        # 保存PyTorch模型
        for task_name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_{task_name}.pt'))
        
        # 保存其他组件
        with open(os.path.join(model_dir, 'num_classes.pkl'), 'wb') as f:
            pickle.dump(self.num_classes, f)
        
        with open(os.path.join(model_dir, 'encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open(os.path.join(model_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open(os.path.join(model_dir, 'features.pkl'), 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"\n[保存] 深度学习模型已保存到: {model_dir}")
    
    def load_models(self, model_dir: str):
        """加载所有模型"""
        # 加载其他组件
        with open(os.path.join(model_dir, 'num_classes.pkl'), 'rb') as f:
            self.num_classes = pickle.load(f)
        
        with open(os.path.join(model_dir, 'encoders.pkl'), 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(os.path.join(model_dir, 'scalers.pkl'), 'rb') as f:
            self.scalers = pickle.load(f)
        
        with open(os.path.join(model_dir, 'features.pkl'), 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        # 重建并加载PyTorch模型
        self.models = {}
        
        for task_name, feature_cols in self.feature_columns.items():
            input_dim = len(feature_cols)
            
            if task_name == 'hours':
                model = MLPRegressor(input_dim=input_dim)
            else:
                num_classes = self.num_classes[task_name]
                model = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
            
            model.load_state_dict(torch.load(
                os.path.join(model_dir, f'model_{task_name}.pt'),
                map_location=torch.device('cpu')
            ))
            model.eval()
            self.models[task_name] = model
        
        self.is_trained = True
        print(f"\n[加载] 深度学习模型已加载: {len(self.models)} 个")


def train_and_save_models_dl(data_file: str = "archive/adult.csv",
                             model_dir: str = "adult_v2/trained_models_dl",
                             epochs: int = 50):
    """训练并保存深度学习模型"""
    print("开始训练判别式辅助模型（深度学习 - PyTorch）...")
    
    models = AdultDiscriminativeModelsDL()
    models.train_all_models(data_file, test_size=0.2, epochs=epochs, verbose=True)
    models.save_models(model_dir)
    
    print("\n训练完成！")
    return models


if __name__ == "__main__":
    train_and_save_models_dl()
