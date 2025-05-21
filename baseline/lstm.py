import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler,RobustScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics_csv import save_metrics_to_csv
# 加载配置参数
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    shared_config = config["shared"]
    lstm_config = config["lstm"]

TIME_WINDOW = shared_config["TIME_WINDOW"]  # 时间窗口长度
BATCH_SIZE = lstm_config["BATCH_SIZE"]
HIDDEN_SIZE = lstm_config["HIDDEN_SIZE"]  # LSTM隐藏层大小
EPOCHS = lstm_config["EPOCHS"]
LR = lstm_config["LR"]


# 数据加载与预处理
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    features = df[shared_config["features"]].values.astype(float)
    scaler = StandardScaler()
    #scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


# 创建时间窗口数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size]
        return x, y


# Tiny-LSTM模型定义
class TinyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步输出
        return out


# 训练流程
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


# 计算混淆矩阵以及其他指标
def compute_metrics(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    tp = cm[1, 1]  # 真阳性
    fp = cm[0, 1]  # 假阳性
    fn = cm[1, 0]  # 假阴性
    tn = cm[0, 0]  # 真阴性

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return cm, accuracy, precision, recall, f1_score





if __name__ == "__main__":
    # 加载训练数据
    train_data, scaler = load_data(shared_config["data"]["train_data"])
    train_dataset = TimeSeriesDataset(train_data, TIME_WINDOW)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 加载验证数据 (使用部分训练数据)
    val_size = int(0.15 * len(train_data))
    val_data = train_data[-val_size:]
    val_dataset = TimeSeriesDataset(val_data, TIME_WINDOW)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 加载测试数据
    test_data, _ = load_data(shared_config["data"]["test_data"])
    test_dataset = TimeSeriesDataset(test_data, TIME_WINDOW)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = TinyLSTM(input_size=3, hidden_size=HIDDEN_SIZE, output_size=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f}")

        # 简单验证
        with torch.no_grad():
            val_loss = 0
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
            val_loss /= len(val_loader)
        #    print(f"Validation Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")

    print("Training completed. Best model saved as best_model.pth")

    # 测试阶段
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # 从测试数据加载标签
    test_df = pd.read_csv(shared_config["data"]["test_data"])
    true_labels = test_df['label'].values
    # 确保标签与测试数据对齐 (预测结果比原始数据少TIME_WINDOW个)
    true_labels = true_labels[TIME_WINDOW : len(test_data)]


    val_errors = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            mse = torch.mean((outputs - batch_y) ** 2, dim=1)
            val_errors.extend(mse.numpy())
    
    # 计算并保存上下限阈值
    threshold_high = 1.0*np.percentile(val_errors, 99.95554)  # 上限阈值(89%分位数)
    threshold_low = np.percentile(val_errors, 30)    # 下限阈值(30%分位数)
    print (f"threshold_high:", threshold_high)
    # 保存双阈值到config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if "lstm" not in config:
        config["lstm"] = {}
    config["lstm"]["THRESHOLD_HIGH"] = float(threshold_high*1)
    config["lstm"]["THRESHOLD_LOW"] = float(threshold_low)
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nLSTM阈值设置:")
    print(f"上限阈值(异常): {threshold_high:.4f}")
    print(f"下限阈值(正常): {threshold_low:.4f}")

    test_labels = np.zeros(len(test_data) - TIME_WINDOW, dtype=int)
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
            outputs = model(batch_x)
            mse = torch.mean((outputs - batch_y) ** 2, dim=1)
            
            # 计算batch在最终结果中的位置
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + len(mse)

            test_labels[start_idx:end_idx] = (mse > threshold_high).int().numpy()
    
    print(f"True labels length: {len(true_labels)}")
    print(f"Predicted labels length: {len(test_labels)}")

    # 调整 true_labels 的长度以匹配 test_labels
    adjusted_true_labels = true_labels[:len(test_labels)]
    cm, accuracy, precision, recall, f1_score = compute_metrics(adjusted_true_labels, test_labels)
    # 打印统计信息
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision, 
        "Recall": recall,
        "F1": f1_score
    }
    save_metrics_to_csv("LSTM", metrics)
    # 打印混淆矩阵
    print("\n混淆矩阵:")
    print("          Predicted")
    print("          Normal Anomaly")
    print(f"Actual Normal  {cm[0,0]:<6} {cm[0,1]:<6}")
    print(f"       Anomaly {cm[1,0]:<6} {cm[1,1]:<6}")
