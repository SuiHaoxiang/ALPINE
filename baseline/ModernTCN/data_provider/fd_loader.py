import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# class FDLoader(Dataset):
#     def __init__(self, root_path='./dataset/data_FD/', win_size=None, flag='train', scale=True, **kwargs):
#         self.root_path = root_path
#         self.scale = scale
#         self.scaler = StandardScaler()
#         self.data = None
#         self.labels = None
        
#     def get_feature_names(self):
#         return ['RSSI', 'Link Quality', 'Ping Delay']
        
#     def _read_data(self, flag='train'):
#         assert flag in ['train', 'test']
#         path = os.path.join(self.root_path, f'{flag}.csv')
#         df = pd.read_csv(path)
        
#         # 删除包含NaN值的行
#         df = df.dropna()
        
#         # 提取特征和标签
#         features = df[['RSSI', 'Link Quality', 'Ping Delay']].values
#         labels = df['label'].values
        
#         # 标准化处理
#         if flag == 'train':
#             self.scaler.fit(features)
#         if self.scale:
#             features = self.scaler.transform(features)
            
#         return features, labels
        
#     def __len__(self):
#         if self.data is None:
#             features, _ = self._read_data('train')
#             self.data = features.reshape(-1, 100, features.shape[-1])  # 假设seq_len=100
#         return len(self.data)
        
#     def __getitem__(self, index):
#         if self.data is None or self.labels is None:
#             features, labels = self._read_data('train')
#             self.data = features.reshape(-1, 100, features.shape[-1])
#             self.labels = labels.reshape(-1, 100)
#         return self.data[index], self.labels[index]
        
#     def __call__(self, args, flag):
#         features, labels = self._read_data(flag)
        
#         # 转换为时间序列格式
#         seq_data = features.reshape(-1, args.seq_len, features.shape[-1])  # [N, L, C]
#         seq_labels = labels.reshape(-1, args.seq_len)  # [N, L]
        
#         # 更新缓存
#         self.data = seq_data
#         self.labels = seq_labels
        
#         data_loader = DataLoader(
#             self,
#             batch_size=args.batch_size,
#             shuffle=(flag=='train'),
#             num_workers=args.num_workers
#         )
#         return self, data_loader

# class FDDataset(Dataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels
        
#     def __len__(self):
#         return len(self.data)
        
#     def __getitem__(self, index):
#         return self.data[index], self.labels[index]

# def data_provider(args, flag):
#     loader = FDLoader(root_path=args.root_path if hasattr(args, 'root_path') else './dataset/data_FD/')
#     return loader(args, flag)
class FD_Loader:
    def __init__(self, root_path, win_size, flag, **kwargs):
        self.root_path = root_path
        self.win_size = win_size
        self.flag = flag
        self.scaler = StandardScaler()
        self.features = ['RSSI', 'Link Quality', 'Ping Delay']
        
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
            df = df.dropna()  # 删除包含NaN值的行
            # 训练数据没有label列
            data = df[self.features].values
            return data, None
        else:  # val/test
            path = os.path.join(self.root_path, 'test.csv')
            df = pd.read_csv(path)
            df = df.dropna()  # 删除包含NaN值的行
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
