"""
Adult Census Data - Holistic Discriminator
整体判别器 - 学习真实数据分布，区分真实样本vs生成样本
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 尝试导入深度学习框架
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


# ============================================================================
#                      整体判别器（传统ML版本）
# ============================================================================

class AdultHolisticDiscriminator:
    """
    整体判别器
    
    核心思想：
    - 学习区分"真实样本"和"生成样本"
    - 不定义因果关系，让模型自己学习整体分布特征
    - 输出真实度评分 [0, 1]，越接近1越像真实数据
    
    用途：
    1. 评估生成样本质量
    2. 过滤低质量样本（真实度评分低）
    3. 提供反馈信号（可选）
    """
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Args:
            model_type: 'random_forest', 'gradient_boosting', 'neural_network'
        """
        self.model_type = model_type
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
    
    def _prepare_sample(self, sample: Dict) -> np.ndarray:
        """将样本转换为特征向量"""
        # 定义所有字段（按Adult数据集标准顺序）
        all_fields = [
            'age', 'workclass', 'fnlwgt', 'education', 'education.num',
            'marital.status', 'occupation', 'relationship', 'race', 'sex',
            'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income'
        ]
        
        if self.feature_columns is None:
            self.feature_columns = all_fields
        
        features = []
        for field in self.feature_columns:
            value = sample.get(field, None)
            
            # 处理缺失值
            if value is None or value == '' or value == '?':
                # 分类字段用特殊标记，数值字段用-1
                if field in ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                           'capital.loss', 'hours.per.week']:
                    features.append(-1)
                else:
                    features.append('__MISSING__')
            else:
                features.append(value)
        
        return np.array(features)
    
    def _encode_features(self, samples: List[Dict], is_training: bool = True) -> np.ndarray:
        """编码特征（分类变量编码+标准化）"""
        # 转换为DataFrame
        df = pd.DataFrame([self._prepare_sample(s) for s in samples], 
                         columns=self.feature_columns)
        
        # 编码分类变量
        for col in self.feature_columns:
            if df[col].dtype == 'object':
                if is_training:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        # 添加__MISSING__到已知类别
                        all_values = df[col].unique().tolist()
                        if '__MISSING__' not in all_values:
                            all_values.append('__MISSING__')
                        self.label_encoders[col].fit(all_values)
                    df[col] = self.label_encoders[col].transform(df[col])
                else:
                    # 处理训练时未见过的类别
                    df[col] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in self.label_encoders[col].classes_
                        else self.label_encoders[col].transform(['__MISSING__'])[0]
                    )
        
        # 转为数值
        X = df.astype(float).values
        
        # 标准化
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X
    
    def train(self, real_samples: List[Dict], 
              synthetic_samples: List[Dict] = None,
              test_size: float = 0.2,
              verbose: bool = True):
        """
        训练整体判别器
        
        Args:
            real_samples: 真实样本
            synthetic_samples: 生成样本（可选，如果没有则用bootstrap采样制造负样本）
            test_size: 测试集比例
            verbose: 是否输出训练信息
        """
        if verbose:
            print("\n" + "=" * 80)
            print("训练整体判别器（Holistic Discriminator）")
            print("=" * 80)
            print(f"\n思想：让模型自动学习真实数据的整体分布特征")
            print(f"无需手动定义因果关系，避免循环依赖问题")
        
        # 准备正样本（真实数据）
        X_real = self._encode_features(real_samples, is_training=True)
        y_real = np.ones(len(X_real))  # 标签1表示真实
        
        # 准备负样本（生成数据或人工制造）
        if synthetic_samples:
            X_fake = self._encode_features(synthetic_samples, is_training=False)
            y_fake = np.zeros(len(X_fake))  # 标签0表示生成
        else:
            # 如果没有生成样本，通过打乱字段制造"假样本"
            if verbose:
                print(f"\n未提供生成样本，使用字段打乱策略制造负样本")
            
            fake_samples = self._create_shuffled_samples(real_samples)
            X_fake = self._encode_features(fake_samples, is_training=False)
            y_fake = np.zeros(len(X_fake))
        
        # 合并数据
        X = np.vstack([X_real, X_fake])
        y = np.hstack([y_real, y_fake])
        
        if verbose:
            print(f"\n训练数据：")
            print(f"  真实样本: {len(X_real)}")
            print(f"  生成样本: {len(X_fake)}")
            print(f"  特征维度: {X.shape[1]}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 训练模型
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, 
                n_jobs=-1, class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                random_state=42, subsample=0.8
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        if verbose:
            print(f"\n训练模型: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        
        # 评估
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]  # 真实度评分
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        if verbose:
            print(f"\n模型性能：")
            print(f"  准确率: {accuracy:.3f}")
            print(f"  AUC: {auc:.3f}")
            
            # 特征重要性（如果模型支持）
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                top_indices = np.argsort(importances)[-10:][::-1]
                print(f"\n最重要的10个特征：")
                for idx in top_indices:
                    print(f"    {self.feature_columns[idx]:20s}: {importances[idx]:.3f}")
        
        self.is_trained = True
        
        if verbose:
            print("\n" + "=" * 80)
            print("训练完成")
            print("=" * 80)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'model_type': self.model_type
        }
    
    def _create_shuffled_samples(self, samples: List[Dict]) -> List[Dict]:
        """通过打乱字段制造假样本（破坏真实分布）"""
        fake_samples = []
        
        # 提取每个字段的所有值
        field_values = {}
        for field in self.feature_columns:
            field_values[field] = [s.get(field) for s in samples]
        
        # 随机组合字段（打乱相关性）
        for i in range(len(samples)):
            fake_sample = {}
            for field in self.feature_columns:
                # 随机选择该字段的一个值
                random_idx = np.random.randint(len(samples))
                fake_sample[field] = field_values[field][random_idx]
            fake_samples.append(fake_sample)
        
        return fake_samples
    
    def score(self, sample: Dict) -> float:
        """
        评估单个样本的真实度
        
        Returns:
            真实度评分 [0, 1]，越接近1越像真实数据
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X = self._encode_features([sample], is_training=False)
        proba = self.model.predict_proba(X)[0, 1]
        
        return float(proba)
    
    def score_batch(self, samples: List[Dict]) -> np.ndarray:
        """批量评估样本真实度"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X = self._encode_features(samples, is_training=False)
        probas = self.model.predict_proba(X)[:, 1]
        
        return probas
    
    def filter_samples(self, samples: List[Dict], 
                      threshold: float = 0.5,
                      verbose: bool = False) -> Tuple[List[Dict], List[float]]:
        """
        过滤低质量样本
        
        Args:
            samples: 待过滤样本
            threshold: 真实度阈值，低于此值的样本被过滤
            verbose: 是否输出统计信息
        
        Returns:
            (保留的样本, 真实度评分)
        """
        scores = self.score_batch(samples)
        
        filtered_samples = []
        filtered_scores = []
        
        for sample, score in zip(samples, scores):
            if score >= threshold:
                filtered_samples.append(sample)
                filtered_scores.append(score)
        
        if verbose:
            print(f"\n[整体判别器] 样本过滤：")
            print(f"  原始样本: {len(samples)}")
            print(f"  保留样本: {len(filtered_samples)}")
            print(f"  过滤率: {(1 - len(filtered_samples)/len(samples)):.1%}")
            print(f"  平均真实度: {np.mean(scores):.3f}")
            print(f"  保留样本平均真实度: {np.mean(filtered_scores):.3f}")
        
        return filtered_samples, filtered_scores
    
    def get_quality_report(self, samples: List[Dict]) -> Dict:
        """生成样本质量报告"""
        scores = self.score_batch(samples)
        
        return {
            'num_samples': len(samples),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'median_score': float(np.median(scores)),
            'high_quality_ratio': float(np.mean(scores >= 0.7)),  # 真实度>0.7的比例
            'low_quality_ratio': float(np.mean(scores < 0.3))   # 真实度<0.3的比例
        }
    
    def save_model(self, model_dir: str):
        """保存模型"""
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'holistic_discriminator.pkl'), 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type
            }, f)
        
        print(f"\n[保存] 整体判别器已保存到: {model_dir}")
    
    def load_model(self, model_dir: str):
        """加载模型"""
        with open(os.path.join(model_dir, 'holistic_discriminator.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.model_type = data['model_type']
        self.is_trained = True
        
        print(f"\n[加载] 整体判别器已加载: {self.model_type}")


# ============================================================================
#                      训练脚本
# ============================================================================

def train_holistic_discriminator(data_file: str = "archive/adult.csv",
                                 model_dir: str = "adult_v2/trained_holistic_discriminator",
                                 model_type: str = 'gradient_boosting',
                                 sample_limit: int = 10000):
    """训练整体判别器"""
    print("开始训练整体判别器...")
    
    # 加载真实数据
    df = pd.read_csv(data_file)
    df = df.replace('?', '__MISSING__')
    
    # 采样（避免数据过大）
    if len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=42)
    
    real_samples = df.to_dict('records')
    
    # 创建并训练判别器
    discriminator = AdultHolisticDiscriminator(model_type=model_type)
    result = discriminator.train(
        real_samples=real_samples,
        synthetic_samples=None,  # 让它自己制造负样本
        verbose=True
    )
    
    # 保存模型
    discriminator.save_model(model_dir)
    
    print("\n训练完成！")
    return discriminator, result


if __name__ == "__main__":
    train_holistic_discriminator()
