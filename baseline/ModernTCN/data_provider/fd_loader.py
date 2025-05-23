import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class FD_Loader:
    def __init__(self, root_path, win_size, flag, **kwargs):
        self.root_path = root_path
        self.win_size = win_size
        self.flag = flag
        self.scaler = StandardScaler()
        self.features = ['RSSI', 'Link Quality', 'Ping Delay']
        
        # Pre-fit scaler during initialization
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

            data = df[self.features].values
            return data, None
        else:  # val/test
            path = os.path.join(self.root_path, 'test.csv')
            df = pd.read_csv(path)
            df = df.dropna() 
            data = df[self.features].values
            labels = df['label'].values if 'label' in df.columns else None
            return data, labels
            
    def _get_freq(self, data):
  
        return 0.1  
        
    def __len__(self):
        data, _ = self._read_data(self.flag)
        return len(data)

    def __getitem__(self, index):
        data, labels = self._read_data(self.flag)
        
        # Ensure scaler is properly initialized
        if self.flag == 'train':
            data = self.scaler.fit_transform(data)
        else:
            if not hasattr(self.scaler, 'mean_'):
                # If scaler not initialized for val/test, initialize with training data
                train_path = os.path.join(self.root_path, 'train.csv')
                train_df = pd.read_csv(train_path)
                train_data = train_df[self.features].values
                self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        
        # Create sliding window samples
        start_idx = index
        end_idx = start_idx + self.win_size
        if end_idx > len(data):
            end_idx = len(data)
            start_idx = end_idx - self.win_size
            
        window_data = data[start_idx:end_idx]
        window_label = labels[start_idx:end_idx] if labels is not None else None
        
        # Convert to tensor
        window_data = torch.FloatTensor(window_data)
        if window_label is not None:
            window_label = torch.FloatTensor(window_label)
        
        return window_data, window_label

    def __call__(self, flag):
        self.flag = flag
        return self
