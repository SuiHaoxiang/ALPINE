import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import yaml
import os
import pickle

def load_config():
    """加载配置文件"""
    with open('/home/code/lstm_if_one_dim/final/src/config.yaml') as f:
        config = yaml.safe_load(f)
    return config

def load_train_data(train_path):
    """加载训练集"""
    print(f"正在加载训练集: {train_path}")
    df = pd.read_csv(train_path)
    # 训练集可能没有标签，使用全0填充
    if 'label' not in df.columns:
        df['label'] = 0
    return df

def load_test_data(test_path):
    """加载测试集""" 
    print(f"正在加载测试集: {test_path}")
    df = pd.read_csv(test_path)
    # 确保测试集有标签列
    if 'label' not in df.columns:
        raise ValueError("测试集必须包含'label'列")
    return df

def preprocess_data(train_data, test_data):
    """数据预处理"""
    # 自动获取数值特征列名（排除label列和非数值列）
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col != 'label']
    
    if not features:
        raise ValueError("没有找到数值特征列")
    
    # 分离特征和标签
    X_train = train_data[features].values.astype(np.float64)
    y_train = train_data['label'].values
    X_test = test_data[features].values.astype(np.float64)
    y_test = test_data['label'].values
    
    # 检查并处理NaN值
    if np.isnan(X_train.astype(np.float64)).any():
        print("警告: 训练数据包含NaN值，将删除包含NaN的行")
        # 获取非NaN行的索引
        non_nan_idx = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[non_nan_idx]
        y_train = y_train[non_nan_idx]
        
    if np.isnan(X_test).any():
        print("警告: 测试数据包含NaN值，将删除包含NaN的行")
        # 获取非NaN行的索引
        non_nan_idx = ~np.isnan(X_test).any(axis=1)
        X_test = X_test[non_nan_idx]
        y_test = y_test[non_nan_idx]
    
    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test

def train_ocsvm(X_train, nu, kernel='rbf', gamma='scale'):
    """训练One-Class SVM模型"""
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X_train)
    
    # 保存训练好的模型
    model_path = '/home/code/lstm_if_one_dim/final/src/svm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到: {model_path}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    # 将One-Class SVM的输出(-1,1)转换为(0,1)
    y_pred = np.where(y_pred == -1, 1, 0)  # -1表示异常，转换为1
    y_true = y_test
    
    # 确保测试集中有异常样本
    if np.sum(y_true) == 0:
        print("警告: 测试集中未发现异常样本")
        return y_pred
    
    # 计算并打印完整分类报告
    print("\n完整分类报告:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    # 计算并打印异常类指标
    try:
        report = classification_report(y_true, y_pred, output_dict=True, digits=4)
        if '1.0' in report:
            anomaly_report = report['1.0']
            print("\n异常检测性能:")
            print(f"精确率(Precision): {anomaly_report['precision']:.4f}")
            print(f"召回率(Recall): {anomaly_report['recall']:.4f}")
            print(f"F1分数: {anomaly_report['f1-score']:.4f}")
    except:
        print("无法计算异常类指标")
    
    # 计算AUC-ROC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"\nAUC-ROC: {auc:.4f}")
    except:
        print("\nAUC-ROC: 需要正负样本才能计算")
    
    return y_pred


def main():
    # 加载配置
    config = load_config()
    
    # 分别加载训练集和测试集
    train_data = load_train_data(config['shared']['data']['train_data'])
    test_data = load_test_data(config['shared']['data']['test_data'])
    
    # 预处理数据
    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
    
    # 训练模型
    model = train_ocsvm(X_train, 
                       nu=0.005,
                       kernel=config.get('svm_kernel', 'rbf'),
                       gamma=config.get('svm_gamma', 'scale'))
    
    # 评估模型
    y_pred = evaluate_model(model, X_test, y_test)
   
if __name__ == "__main__":
    main()
