import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # 原有实现...
        pass

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # 原有实现...
        pass

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='custom.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # 原有实现...
        pass

class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='solar_AL.txt',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # 原有实现...
        pass

class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='PEMS04.npz',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # 原有实现...
        pass

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='custom.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # 原有实现...
        pass

class Dataset_SD(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path=None,
                 target='label', scale=True, timeenc=0, freq='h'):
        # SD数据集特定实现
        self.required_cols = ['signal_strength','transmit_rate','ping_latency']
        self.root_path = root_path
        self.flag = flag
        self.size = size
        self.features = features
        self.data_path = data_path
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)
        print(f"Loading SD data from: {file_path}")
        
        df_raw = pd.read_csv(file_path)
        print("Columns found:", df_raw.columns.tolist())
        
        # 去除NaN值
        initial_count = len(df_raw)
        df_raw = df_raw.dropna()
        removed_count = initial_count - len(df_raw)
        if removed_count > 0:
            print(f"Removed {removed_count} rows containing NaN values")
        
        # 检查特征列
        feature_cols = ['signal_strength','transmit_rate','ping_latency']
        missing_feats = [col for col in feature_cols if col not in df_raw.columns]
        if missing_feats:
            raise ValueError(f"SD数据集缺少必需特征列: {missing_feats}")

        # 处理标签列（可选）
        df_data = df_raw[feature_cols]
        if 'label' in df_raw.columns:
            df_label = df_raw['label'].astype(int)
            if not set(df_label.unique()).issubset({0, 1}):
                raise ValueError("标签必须是0(正常)和1(异常)的二分类值")
        else:
            df_label = pd.Series(0, index=df_raw.index)  # 默认所有数据正常
        
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        data_stamp = np.zeros((len(df_raw), 1))  # SD数据集不使用时间特征

        self.data_x = data
        self.data_y = df_label.values
        self.data_stamp = data_stamp
        print("SD data loaded successfully. Shape:", self.data_x.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.size[0]
        
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = np.zeros_like(seq_x_mark)
        
        seq_x = torch.from_numpy(seq_x).float()
        seq_x_mark = torch.from_numpy(seq_x_mark).float()
        seq_y_mark = torch.from_numpy(seq_y_mark).float()

        if self.flag == 'train':
            batch_y = seq_x.unsqueeze(0) if seq_x.dim() == 1 else seq_x
            return seq_x, batch_y, seq_x_mark, seq_y_mark
        else:
            seq_y = torch.tensor(self.data_y[s_end-1]).float()
            return seq_x, seq_y.view(1,1,1), seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.size[0] + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_UCR(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='label', scale=True, timeenc=0, freq='h'):
        # UCR数据集特殊处理
        self.flag = flag  # 添加flag属性
        self.size = size
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)
        print(f"Loading UCR data from: {file_path}")
        
        # 加载npy文件
        data = np.load(file_path)
        print(f"Loaded UCR data shape: {data.shape}")
        
        # 创建默认标签(全0)
        labels = np.zeros(len(data))
        
        if self.scale:
            self.scaler.fit(data)
            data = self.scaler.transform(data)
            
        # 不使用时间特征
        data_stamp = np.zeros((len(data), 1))

        self.data_x = data
        self.data_y = labels
        self.data_stamp = data_stamp
        print("UCR data loaded successfully. Shape:", self.data_x.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.size[0]
        
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = np.zeros_like(seq_x_mark)
        
        seq_x = torch.from_numpy(seq_x).float()
        seq_x_mark = torch.from_numpy(seq_x_mark).float()
        seq_y_mark = torch.from_numpy(seq_y_mark).float()

        if self.flag == 'train':
            batch_y = seq_x.unsqueeze(0) if seq_x.dim() == 1 else seq_x
            return seq_x, batch_y, seq_x_mark, seq_y_mark
        else:
            seq_y = torch.tensor(self.data_y[s_end-1]).float()
            return seq_x, seq_y.view(1,1,1), seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.size[0] + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_FD(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path=None,
                 target='label', scale=True, timeenc=0, freq='h'):
        # 根据flag自动选择数据文件
        if data_path is None:
            data_path = 'train.csv' if flag == 'train' else 'test.csv'
        self.flag = flag  # 添加flag属性
        self.timeenc = 0  # 固定为0，不使用时间特征
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.features = features
        self.target = target
        self.scale = scale
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)
        print(f"Loading data from: {file_path}")
        
        try:
            df_raw = pd.read_csv(file_path)
            print("Columns found:", df_raw.columns.tolist())
            
            # 动态识别特征列（排除Timestamp和label）
            feature_cols = [col for col in df_raw.columns if col not in ['Timestamp', 'label']]
            if len(feature_cols) < 1:
                raise ValueError("数据集必须包含至少1个特征列")
            
            # 确保标签是0/1二分类
            df_data = df_raw[feature_cols]
            df_label = df_raw['label'].astype(int)  # 强制转换为整数
            if not set(df_label.unique()).issubset({0, 1}):
                raise ValueError("标签必须是0(正常)和1(异常)的二分类值")
            
            if self.scale:
                self.scaler.fit(df_data.values)
                data = self.scaler.transform(df_data.values).astype(np.float32)  # 确保转换为float32
            else:
                data = df_data.values.astype(np.float32)  # 确保转换为float32
                
            # 不使用时间特征，创建空的时间标记
            data_stamp = np.zeros((len(df_raw), 1), dtype=np.float32)  # 确保为float32

            self.data_x = data
            self.data_y = df_label.values.astype(np.float32)  # 确保标签也是float32
            self.data_stamp = data_stamp
            print("Data loaded successfully. Shape:", self.data_x.shape)

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end]  # [seq_len, 3]
        seq_x_mark = self.data_stamp[s_begin:s_end]  # [seq_len, time_features]
        seq_y_mark = np.zeros_like(seq_x_mark)  # 添加空y_mark
        
        # 将numpy数组转换为PyTorch张量
        seq_x = torch.from_numpy(seq_x).float()
        seq_x_mark = torch.from_numpy(seq_x_mark).float()
        seq_y_mark = torch.from_numpy(seq_y_mark).float()

        # 异常检测任务，使用序列重构误差作为目标
        if self.flag == 'train':
            # 训练时返回输入序列作为目标(自监督学习)
            # 确保返回的batch_y是3D张量 [batch_size, seq_len, features]
            batch_y = seq_x.unsqueeze(0) if seq_x.dim() == 1 else seq_x
            return seq_x, batch_y, seq_x_mark, seq_y_mark
        else:
            # 测试时返回实际标签(如果有)
            # 确保返回的batch_y是3D张量 [batch_size, 1, 1]
            seq_y = torch.tensor(self.data_y[s_end-1]).float()  # 取最后一个时间点的label
            return seq_x, seq_y.view(1,1,1), seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
