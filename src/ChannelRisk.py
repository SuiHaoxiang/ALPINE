import pandas as pd
import numpy as np
import torch
import pickle
import yaml
from pathlib import Path
import sys
from lstm import TinyLSTM
from spa import SparseIsolationForest,IsolationTree,IsolationTreeNode
from sklearn.preprocessing import StandardScaler

class ChannelRisk:
    """通道风险计算类，封装风险计算逻辑"""

    def __init__(self, config_path="config.yaml"):
        """初始化风险计算器

        Args:
            config_path (str): 配置文件路径
        """
        # 加载配置
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            hyb_config = config["hyb"]
            shared_config = config["shared"]
            lstm_config = config["lstm"]

        # 获取配置参数
        self.TIME_WINDOW = shared_config["TIME_WINDOW"]
        self.HIDDEN_SIZE = lstm_config["HIDDEN_SIZE"]
        self.LSTM_TH = lstm_config["THRESHOLD_HIGH"]
        self.GLOBAL_HIGH_THRESHOLD = config["spa"]["GLOBAL_THRESHOLD"]
        self.GLOBAL_LOW_THRESHOLD = config["spa"]["NORMAL_THRESHOLD"]
        self.features = shared_config["features"]
        
        # 加载测试数据用于标准化器拟合
        test_df = pd.read_csv(shared_config["data"]["test_data"])
        
        # 初始化标准化器
        self.scaler = StandardScaler()
        self.scaler.fit(test_df[self.features].values.astype(float))
        
        # 加载模型
        with open(hyb_config["models"]["sif_model"], "rb") as f:
            self.sif_model = pickle.load(f)

        self.lstm_model = TinyLSTM(
            input_size=3, 
            hidden_size=self.HIDDEN_SIZE, 
            output_size=3
        )
        self.lstm_model.load_state_dict(
            torch.load(hyb_config["models"]["lstm_model"])
        )
        self.lstm_model.eval()
        
        # 初始化历史数据缓冲区
        self.history_buffer = []
    
    def calculate_risk(self, current_features):
        """计算当前特征的风险值
        
        Args:
            current_features (np.array): 当前时间步的特征数组
            
        Returns:
            tuple: (风险值, 决策原因)
        """
        # 标准化当前特征
        scaled_features = self.scaler.transform(
            current_features.reshape(1, -1).astype(float))
        sif_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        
        # 计算Sparse-iForest分数
        s_score = self.sif_model.anomaly_score(sif_tensor).item()
        
        # 第一层判断
        if s_score > self.GLOBAL_HIGH_THRESHOLD:
            return 1.0, "SIF-Anomaly"
        elif s_score < self.GLOBAL_LOW_THRESHOLD:
            return 0.1, "SIF-Normal"
        
        # 第二层LSTM判断
        if len(self.history_buffer) >= self.TIME_WINDOW:
            # 获取时间窗口数据
            window_data = np.array(self.history_buffer[-self.TIME_WINDOW:])
            scaled_window = self.scaler.transform(window_data.astype(float))
            lstm_input = torch.FloatTensor(scaled_window).unsqueeze(0)
            
            # LSTM预测
            with torch.no_grad():
                pred = self.lstm_model(lstm_input)
                true = torch.FloatTensor(scaled_features)
                delta = torch.mean((pred - true) ** 2).item()
            
            if delta > self.LSTM_TH:
                return 1.0, "LSTM-Anomaly"
            else:
                return 0.1, "LSTM-Judged"
        else:
            return 1, "Insufficient-History"
    
    def update_history(self, current_features):
        """更新历史数据缓冲区
        
        Args:
            current_features (np.array): 当前时间步的特征数组
        """
        self.history_buffer.append(current_features)
    
    def process_sample(self, features):
        """处理单个样本
        
        Args:
            features (np.array): 当前样本的特征数组
            
        Returns:
            tuple: (风险值, 决策原因)
        """
        self.update_history(features)
        return self.calculate_risk(features)

# 示例用法
if __name__ == "__main__":
    # 初始化风险计算器
    risk_calculator = ChannelRisk()
    
    # 加载测试数据
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        shared_config = config["shared"]
    test_df = pd.read_csv(shared_config["data"]["test_data"])
    
    # 单个样本处理示例
    for i in range(len(test_df)):
        features = test_df[shared_config["features"]].iloc[i].values
        risk, reason = risk_calculator.process_sample(features)
        print(f"Sample {i}: {risk:.1f} ({reason})")
