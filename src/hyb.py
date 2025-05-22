import pandas as pd
import numpy as np
import torch
import pickle
import yaml
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from lstm import TinyLSTM, TimeSeriesDataset, load_data as load_lstm_data
from spa import SparseIsolationForest, IsolationTree, IsolationTreeNode
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os

sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics_csv import save_metrics_to_csv
# 加载配置参数
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    hyb_config = config["hyb"]
    shared_config = config["shared"]
    lstm_config = config["lstm"]
TIME_WINDOW = shared_config["TIME_WINDOW"]
HIDDEN_SIZE = lstm_config["HIDDEN_SIZE"]
# 直接从config获取阈值
LSTM_TH = config["lstm"]["THRESHOLD_HIGH"]
GLOBAL_HIGH_THRESHOLD = config["spa"]["GLOBAL_THRESHOLD"]
GLOBAL_LOW_THRESHOLD = config["spa"]["NORMAL_THRESHOLD"]
# 加载测试数据（原始数据）
test_df = pd.read_csv(shared_config["data"]["test_data"])
true_labels = test_df['label'].values

# ================== 数据预处理 ==================
# 加载全量数据拟合scaler（仅用训练数据更合理，此处保持原有逻辑）
full_data = pd.read_csv(shared_config["data"]["test_data"])
scaler = StandardScaler()
scaler.fit(full_data[shared_config["features"]].values.astype(float))

# 初始化存储
final_pred = np.zeros(len(test_df), dtype=int)  # 最终预测结果
lstm_judge_count = 0  # LSTM判断计数器
sif_normal_count = 0  # Sparse-iForest判定正常样本计数
sif_anomaly_count = 0  # Sparse-iForest判定异常样本计数
sif_anomaly_indices = []  # 记录Sparse-iForest判定的异常索引
sif_scores = np.zeros(len(test_df))  # 记录Sparse-iForest分数
lstm_scores = np.zeros(len(test_df))  # 记录LSTM MSE分数

# ================== 加载模型 ==================
# 加载Sparse-iForest模型
with open(hyb_config["models"]["sif_model"], "rb") as f:
    sif_model = pickle.load(f)

# 加载LSTM模型
lstm_model = TinyLSTM(input_size=3, hidden_size=HIDDEN_SIZE, output_size=3)
lstm_model.load_state_dict(torch.load(hyb_config["models"]["lstm_model"]))
lstm_model.eval()

# ================== 逐样本处理 ==================
for i in range(TIME_WINDOW, len(test_df)):
    # ------------------ Sparse-iForest检测 ------------------
    # 提取当前样本特征（注意保持二维形状）
    current_features = test_df[shared_config["features"]].iloc[i].values.reshape(1, -1)
    # ==== 新增标准化处理 ====
    scaled_features = scaler.transform(current_features.astype(float))  # 标准化处理

    sif_tensor = torch.tensor(scaled_features, dtype=torch.float32)  # 使用标准化后数据

    # 计算并记录全局异常分数
    global_score = sif_model.anomaly_score(sif_tensor).item()  # 标量值
    sif_scores[i] = global_score  # 记录Sparse-iForest分数

    # 初步判断
    if global_score > GLOBAL_HIGH_THRESHOLD:
        final_pred[i] = 1
        sif_anomaly_count += 1  # Sparse-iForest判定为异常
        sif_anomaly_indices.append(i)  # 记录异常索引
        continue  # 直接标记为异常，跳过LSTM
    elif global_score < GLOBAL_LOW_THRESHOLD:
        final_pred[i] = 0
        sif_normal_count += 1  # Sparse-iForest判定为正常
        continue  # 直接标记为正常，跳过LSTM

    # ------------------ LSTM检测 ------------------
    # 提取时间窗口数据（i-TIME_WINDOW到i-1）
    window_data = test_df[shared_config["features"]].iloc[i - TIME_WINDOW:i].values

    # 标准化处理（使用预训练的scaler）
    scaled_window = scaler.transform(window_data.astype(float))

    # 转换为张量并添加批次维度
    lstm_input = torch.FloatTensor(scaled_window).unsqueeze(0)  # shape: [1, TIME_WINDOW, features]

    # 预测与误差计算
    with torch.no_grad():
        pred = lstm_model(lstm_input)  # 预测下一时刻
        true = torch.FloatTensor(scaler.transform(test_df[shared_config["features"]].iloc[i].values.reshape(1, -1)))
        mse = torch.mean((pred - true) ** 2).item()

    # 判断LSTM结果并记录分数
    final_pred[i] = 1 if mse > LSTM_TH else 0
    lstm_scores[i] = mse  # 记录LSTM MSE分数
    lstm_judge_count += 1

# ================== 评估与输出 ==================
# 注意结果从TIME_WINDOW开始有效
valid_pred = final_pred[TIME_WINDOW:]
valid_true = true_labels[TIME_WINDOW:]

# 计算各模型独立预测结果
sif_only_pred = np.zeros_like(valid_true)
lstm_only_pred = np.zeros_like(valid_true)

for i in range(len(valid_pred)):
    idx = i + TIME_WINDOW
    if idx in sif_anomaly_indices:
        sif_only_pred[i] = 1
    elif final_pred[idx] == 1:
        lstm_only_pred[i] = 1

# 计算各模型指标
def print_metrics(name, true, pred):
    acc = accuracy_score(true, pred)
    prec = precision_score(true, pred)
    rec = recall_score(true, pred)
    f1_val = f1_score(true, pred)
    cm = confusion_matrix(true, pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n=== {name} 模型性能 ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}") 
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1_val:.4f}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print("\n--- 混淆矩阵 ---")
    print(f"{'True/Pred':<10} {'Normal':<8} {'Anomaly':<8}")
    print(f"{'Normal':<10} {tn:<8} {fp:<8}")
    print(f"{'Anomaly':<10} {fn:<8} {tp:<8}") 
# 打印各模型独立性能
print_metrics("Sparse-iForest", valid_true, sif_only_pred)
print_metrics("LSTM", valid_true, lstm_only_pred)

# 计算混合模型指标
accuracy = accuracy_score(valid_true, valid_pred)
precision = precision_score(valid_true, valid_pred)
recall = recall_score(valid_true, valid_pred) 
f1 = f1_score(valid_true, valid_pred)

# 计算Sparse-iForest判定为正常的样本数中实际异常的数量
sif_normal_indices = [i for i in range(len(final_pred)) if final_pred[i] == 0 and i not in sif_anomaly_indices]
sif_normal_actual_anomalies = sum(true_labels[idx] for idx in sif_normal_indices)

print("\n===== 逐样本处理结果 =====")
print(f"总样本数: {len(valid_true)}")
print(f"真实异常数: {sum(valid_true)}")
print(f"预测异常数: {sum(valid_pred)}")
print(f"直接判为异常的样本数: {np.sum(final_pred == 1)}")
print(f"直接判为异常: {np.sum(final_pred == 1) - lstm_judge_count}")
print(f"直接判为正常: {np.sum(final_pred == 0) - lstm_judge_count}")
print(f"LSTM判断次数: {lstm_judge_count}")
print(f"Sparse-iForest判定为异常的样本数: {sif_anomaly_count}")
print(f"Sparse-iForest判定为正常的样本数: {sif_normal_count}")
print(f"Sparse-iForest判定为正常的样本数中异常的有: {sif_normal_actual_anomalies}")
print(f"最终异常数: {np.sum(valid_pred)}")

# 后续指标保存与输出（保持原有代码不变）
# ================== 评估指标计算 ==================
# 确保对齐有效索引
assert len(valid_pred) == len(valid_true), "预测结果与真实标签长度不一致"

# 计算详细指标
accuracy = accuracy_score(valid_true, valid_pred)
precision = precision_score(valid_true, valid_pred)
recall = recall_score(valid_true, valid_pred)
f1 = f1_score(valid_true, valid_pred)
cm = confusion_matrix(valid_true, valid_pred)

# 扩展指标显示
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)  # 误报率
fnr = fn / (fn + tp)  # 漏报率

# ================== 控制台输出 ==================
print("\n" + "=" * 30 + " 详细评估结果 " + "=" * 30)
print(f"总样本数: {len(valid_true)}")
print(f"真实异常数: {sum(valid_true)}")
print(f"预测异常数: {sum(valid_pred)}")
print(f"LSTM判断次数: {lstm_judge_count}")
print(f"Sparse-iForest判定为异常的样本数: {sif_anomaly_count}")
print(f"Sparse-iForest判定为正常的样本数: {sif_normal_count}")
print("\n--- 分类指标 ---")
print(f"{'Accuracy:':<12} {accuracy:.4f}")
print(f"{'Precision:':<12} {precision:.4f}")
print(f"{'Recall:':<12} {recall:.4f}")
print(f"{'F1 Score:':<12} {f1:.4f}")
print(f"{'FPR:':<12} {fpr:.4f}")
print(f"{'FNR:':<12} {fnr:.4f}")

# 打印模型分数统计
print("\n=== 模型分数统计 ===")
valid_sif_scores = sif_scores[TIME_WINDOW:]
valid_lstm_scores = lstm_scores[TIME_WINDOW:]

print(f"Sparse-iForest分数统计:")
print(f"- 最小值: {valid_sif_scores.min():.4f}")
print(f"- 最大值: {valid_sif_scores.max():.4f}") 
print(f"- 平均值: {valid_sif_scores.mean():.4f}")
print(f"- 异常阈值: {GLOBAL_HIGH_THRESHOLD:.4f}")

print(f"\nLSTM MSE分数统计:")
print(f"- 最小值: {valid_lstm_scores[valid_lstm_scores > 0].min():.4f}")
print(f"- 最大值: {valid_lstm_scores.max():.4f}")
print(f"- 平均值: {valid_lstm_scores[valid_lstm_scores > 0].mean():.4f}")
print(f"- 异常阈值: {LSTM_TH:.4f}")

print("\n--- 混淆矩阵 ---")
print(f"{'True/Pred':<10} {'Normal':<8} {'Anomaly':<8}")
print(f"{'Normal':<10} {tn:<8} {fp:<8}")
print(f"{'Anomaly':<10} {fn:<8} {tp:<8}") 

# ================== 指标保存 ==================
metrics = {
    "Accuracy": round(accuracy, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1": round(f1, 4),
    "FPR": round(fpr, 4),
    "FNR": round(fnr, 4),
    "TP": int(tp),
    "FP": int(fp),
    "TN": int(tn),
    "FN": int(fn)
}
metrics_hyb={
     "Accuracy": round(accuracy, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1": round(f1, 4),

}
params = {
    "TIME_WINDOW": TIME_WINDOW,
    "LSTM_TH": LSTM_TH,
    "GLOBAL_HIGH_THRESHOLD": GLOBAL_HIGH_THRESHOLD,
    "GLOBAL_LOW_THRESHOLD": GLOBAL_LOW_THRESHOLD
}
#save_metrics_to_csv("Hybrid", metrics_hyb)

