import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
import yaml
import os
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, train_loader, config):
    """训练自动编码器"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('ae_lr', 0.001))
    
    model.train()
    for epoch in range(config.get('ae_epochs', 100)):
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')
    
    # 保存训练好的模型
    model_path = config.get('model_path', 'best_autoencoder_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return model

def detect_anomalies(model, data_loader, threshold):
    """检测异常点"""
    model.eval()
    anomalies = []
    with torch.no_grad():
        for data in data_loader:
            outputs = model(data)
            mse = torch.mean((outputs - data)**2, dim=1)
            anomalies.extend((mse > threshold).numpy())
    return np.array(anomalies)

def evaluate(y_true, error_scores, target_recall=None, quantile=0.60):
    """评估模型性能并选择阈值
    参数:
        y_true: 真实标签
        error_scores: 误差分数
        target_recall: 目标召回率
        quantile: 异常百分位数(如0.60表示60%分位数)
    """
    # 打印误差分数分布信息
    print("\n误差分数统计:")
    print(f"最小值: {np.min(error_scores):.4f}")
    print(f"25%分位数: {np.percentile(error_scores, 25):.4f}")
    print(f"中位数: {np.median(error_scores):.4f}") 
    print(f"75%分位数: {np.percentile(error_scores, 75):.4f}")
    print(f"95%分位数: {np.percentile(error_scores, 95):.4f}")
    print(f"最大值: {np.max(error_scores):.4f}")
    # 如果指定了百分位数，直接计算阈值
    if quantile is not None:
        threshold = np.percentile(error_scores, quantile * 100)
        preds = (error_scores > threshold).astype(int)
        f1 = f1_score(y_true, preds)
        print(f"\n使用{quantile*100:.0f}%分位数作为阈值:")
        print(f"阈值: {threshold:.4f}")
        precision = precision_score(y_true, preds)
        recall = recall_score(y_true, preds)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        return threshold
        
    print("\n异常检测性能:")
    
    # 计算AUC-ROC
    fpr, tpr, _ = roc_curve(y_true, error_scores)
    print(f"AUC-ROC: {auc(fpr, tpr):.4f}")
    
    # 计算precision-recall曲线
    precision, recall, thresholds = precision_recall_curve(y_true, error_scores)
    
    # 找到满足目标召回率的最小阈值
    target_idx = np.where(recall >= target_recall)[0]
    if len(target_idx) > 0:
        best_idx = target_idx[0]
        best_threshold = thresholds[best_idx]
        best_precision = precision[best_idx]
        print(f"当召回率>={target_recall}时:")
        print(f"最佳阈值: {best_threshold:.4f}")
        print(f"Precision: {best_precision:.4f}")
        print(f"Recall: {recall[best_idx]:.4f}")
        preds = (error_scores > best_threshold).astype(int)
        f1 = f1_score(y_true, preds)
        print(f"F1分数: {f1:.4f}")
    else:
        print(f"无法达到目标召回率{target_recall}")
    
    return best_threshold if 'best_threshold' in locals() else None

def load_data(file_path):
    """加载并预处理数据"""
    df = pd.read_csv(file_path)
    df = df.dropna()
    return df

def main():
    # 加载配置
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # 数据加载和预处理
    train_data = load_data(config['shared']['data']['train_data'])
    test_data = load_data(config['shared']['data']['test_data'])
    
    # 预处理
    features = config['shared']['features']
    X_train = train_data[features].values
    X_test = test_data[features].values
    y_test = test_data['label'].values
    
    # 检查特征是否存在
    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        raise ValueError(f"配置文件中指定的特征不存在于数据中: {missing_features}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 转换为PyTorch张量
    train_tensor = torch.FloatTensor(X_train)
    test_tensor = torch.FloatTensor(X_test)
    
    # 创建模型
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, encoding_dim=config.get('ae_encoding_dim', 8))
    
    # 训练模型
    model = train_autoencoder(model, [train_tensor], config)
    
    # 计算测试集重构误差
    with torch.no_grad():
        test_recon = model(test_tensor)
        test_errors = torch.mean((test_recon - test_tensor)**2, dim=1).numpy()
    
    # 打印当前配置
    print("\n当前配置参数:")
    print(f"ae_threshold_quantile: {config['autoencoder'].get('ae_threshold_quantile')}")
    
    # 评估并获取最佳阈值
    best_threshold = evaluate(y_test, test_errors,
                            quantile=config['autoencoder'].get('ae_threshold_quantile'))
    
    if best_threshold is not None:
        # 使用最佳阈值检测异常
        test_anomalies = (test_errors > best_threshold).astype(int)
        print("\n使用最佳阈值的检测结果:")
        print(f"异常样本数: {sum(test_anomalies)}")
        print(f"正常样本数: {len(test_anomalies) - sum(test_anomalies)}")

if __name__ == "__main__":
    main()
