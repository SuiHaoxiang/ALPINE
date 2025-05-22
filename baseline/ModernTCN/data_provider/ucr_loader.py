import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class UCRDataset(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # Load data files
        train_data = np.load(os.path.join(root_path, 'train.npy'))
        test_data = np.load(os.path.join(root_path, 'test.npy'))
        labels = np.load(os.path.join(root_path, 'labels.npy'))
        
        # Ensure data is 3D (samples, timesteps, features)
        if train_data.ndim == 2:
            train_data = train_data[:, :, np.newaxis]
        if test_data.ndim == 2:
            test_data = test_data[:, :, np.newaxis]
            
        # For anomaly detection, train only on normal samples (label=0)
        if flag == "train":
            normal_idx = np.where(labels == 0)[0]
            train_data = train_data[normal_idx]
            labels = labels[normal_idx]
            
        # Fit scaler on training data
        self.scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
        
        # Transform data
        self.train = self.scaler.transform(
            train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
        self.test = self.scaler.transform(
            test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
        
        self.labels = labels
        # For anomaly detection:
        # - Train: only normal samples from train.npy
        # - Val/Test: all samples from test.npy
        self.val = self.test  
        self.test = self.test
        
    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            x = np.float32(self.train[index:index+self.win_size])
            y = np.float32(self.labels[0:self.win_size])
            return torch.from_numpy(x), torch.from_numpy(y)  # [seq_len] for univariate
        elif self.flag == 'val':
            x = np.float32(self.val[index:index+self.win_size])
            y = np.float32(self.labels[0:self.win_size])
            return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y)
        elif self.flag == 'test':
            x = np.float32(self.test[index:index+self.win_size])
            y = np.float32(self.labels[index:index+self.win_size])
            return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y)
        else:
            x = np.float32(self.test[index//self.step*self.win_size:index//self.step*self.win_size+self.win_size])
            y = np.float32(self.labels[index//self.step*self.win_size:index//self.step*self.win_size+self.win_size])
            return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y)
