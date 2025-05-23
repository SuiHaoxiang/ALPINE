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

# Load configuration parameters
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    shared_config = config["shared"]
    lstm_config = config["lstm"]

TIME_WINDOW = shared_config["TIME_WINDOW"]  # Time window length
BATCH_SIZE = lstm_config["BATCH_SIZE"]
HIDDEN_SIZE = lstm_config["HIDDEN_SIZE"]  # LSTM hidden layer size
EPOCHS = lstm_config["EPOCHS"]
LR = lstm_config["LR"]


# Data loading and preprocessing
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    features = df[shared_config["features"]].values.astype(float)
    scaler = StandardScaler()
    #scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


# Create time window dataset
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


# Tiny-LSTM model definition
class TinyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take output from last timestep
        return out


# Training process
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


# Compute confusion matrix and other metrics
def compute_metrics(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    tp = cm[1, 1]  # True positive
    fp = cm[0, 1]  # False positive
    fn = cm[1, 0]  # False negative
    tn = cm[0, 0]  # True negative

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return cm, accuracy, precision, recall, f1_score
