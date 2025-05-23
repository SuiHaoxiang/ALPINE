import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class UCRCSVDataset(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.scaler_path = os.path.join(root_path, 'scaler.pkl')
        
        # Load CSV files
        if flag == "train":
            df = pd.read_csv(os.path.join(root_path, 'train.csv'))
            # Handle different column names
            value_col = 'value' if 'value' in df.columns else df.columns[0]
            label_col = 'label' if 'label' in df.columns else None
            
            if label_col:
                # For anomaly detection, train only on normal samples (label=0)
                normal_df = df[df[label_col] == 0]
                self.data = normal_df[value_col].values.reshape(-1, 1)
                self.labels = normal_df[label_col].values
            else:
                # If no label column, assume all are normal samples
                self.data = df[value_col].values.reshape(-1, 1)
                self.labels = np.zeros(len(self.data))
        else:  # test/val
            # Use separate test.csv for evaluation
            df = pd.read_csv(os.path.join(root_path, 'test.csv'))
            value_col = 'value' if 'value' in df.columns else df.columns[0]
            label_col = 'label' if 'label' in df.columns else None
            
            self.data = df[value_col].values.reshape(-1, 1)
            self.labels = df[label_col].values if label_col else np.zeros(len(self.data))
            
        # Handle scaler fit/transform
        if flag == "train":
            self.scaler.fit(self.data)
            # Save scaler for test/val
            import joblib
            joblib.dump(self.scaler, self.scaler_path)
            self.data = self.scaler.transform(self.data)
        else:
            # Load scaler from training
            import joblib
            self.scaler = joblib.load(self.scaler_path)
            self.data = self.scaler.transform(self.data)
        
        # Convert to 3D (1, seq_len, 1) for single sample
        self.data = self.data.reshape(1, -1, 1)
        self.labels = self.labels.reshape(1, -1)

    def __len__(self):
        seq_len = self.data.shape[1]
        if self.flag == "train":
            return (seq_len - self.win_size) // self.step + 1
        else:  # test/val
            return (seq_len - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        x = np.float32(self.data[0, index:index+self.win_size, :])  # (win_size, 1)
        y = np.float32(self.labels[0, index:index+self.win_size])   # (win_size,)
        
        # Reshape output to (1, win_size, 1)
        x = x.reshape(1, -1, 1)  # Add batch dimension
        
        # Ensure correct output shape
        assert x.shape == (1, self.win_size, 1), f"Data shape should be (1, {self.win_size}, 1), got {x.shape}"
        #print(f"Data sample shape: {x.shape}")
        return torch.from_numpy(x), torch.from_numpy(y)
