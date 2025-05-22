import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

class SD_Loader:
    def __init__(self, root_path, win_size, flag, **kwargs):
        self.root_path = root_path
        self.win_size = win_size
        self.flag = flag
        self.scaler = StandardScaler()
        self.features = ['signal_strength', 'transmit_rate', 'ping_latency']
        
        # 在初始化时预先fit scaler
        if flag == 'train':
            path = os.path.join(self.root_path, 'train.csv')
            df = pd.read_csv(path)
            data = df[self.features].values
            self.scaler.fit(data)
        
    def _read_data(self, flag):
        if flag == 'train':
            path = os.path.join(self.root_path, 'train.csv')
            df = pd.read_csv(path)
            df = df.dropna()
            # 训练时取前80%作为训练集
            train_size = int(0.8 * len(df))
            data = df[self.features].values[:train_size]
            return data, None
        elif flag == 'val':  # 验证集用训练集的后20%
            path = os.path.join(self.root_path, 'train.csv')
            df = pd.read_csv(path)
            df = df.dropna()
            val_size = int(0.8 * len(df))
            data = df[self.features].values[val_size:]
            return data, None
        else:  # test
            path = os.path.join(self.root_path, 'test.csv')
            df = pd.read_csv(path)
            df = df.dropna()
            data = df[self.features].values
            labels = df['label'].values if 'label' in df.columns else None
            return data, labels
            
    def _get_freq(self, data):
        # 计算采样频率(Hz)
        return 0.1  # 10秒间隔=0.1Hz
        
    def __len__(self):
        data, _ = self._read_data(self.flag)
        return len(data)

    def __getitem__(self, index):
        data, labels = self._read_data(self.flag)
        
        # 确保scaler已被正确初始化
        if self.flag == 'train':
            data = self.scaler.fit_transform(data)
        else:
            if not hasattr(self.scaler, 'mean_'):
                # 如果验证/测试时scaler未初始化，使用训练数据初始化
                train_path = os.path.join(self.root_path, 'train.csv')
                train_df = pd.read_csv(train_path)
                train_data = train_df[self.features].values
                self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        
        # 创建滑动窗口样本
        start_idx = index
        end_idx = start_idx + self.win_size
        if end_idx > len(data):
            end_idx = len(data)
            start_idx = end_idx - self.win_size
            
        window_data = data[start_idx:end_idx]
        window_label = labels[start_idx:end_idx] if labels is not None else None
        
        # 转换为tensor
        window_data = torch.FloatTensor(window_data)
        if window_label is not None:
            window_label = torch.FloatTensor(window_label)
        
        return window_data, window_label

    def __call__(self, flag):
        self.flag = flag
        return self
