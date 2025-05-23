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
        out = self.fc(out[:, -1, :])  # Take last timestep output
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


# Calculate confusion matrix and other metrics
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





if __name__ == "__main__":
    # Load training data
    train_data, scaler = load_data(shared_config["data"]["train_data"])
    train_dataset = TimeSeriesDataset(train_data, TIME_WINDOW)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load validation data (using part of training data)
    val_size = int(0.15 * len(train_data))
    val_data = train_data[-val_size:]
    val_dataset = TimeSeriesDataset(val_data, TIME_WINDOW)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Load test data
    test_data, _ = load_data(shared_config["data"]["test_data"])
    test_dataset = TimeSeriesDataset(test_data, TIME_WINDOW)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = TinyLSTM(input_size=3, hidden_size=HIDDEN_SIZE, output_size=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f}")

        # Simple validation
        with torch.no_grad():
            val_loss = 0
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
            val_loss /= len(val_loader)
        #    print(f"Validation Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")

    print("Training completed. Best model saved as best_model.pth")

    # Testing phase
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # Load labels from test data
    test_df = pd.read_csv(shared_config["data"]["test_data"])
    true_labels = test_df['label'].values
    # Align labels with test data (predictions are TIME_WINDOW shorter than original data)
    true_labels = true_labels[TIME_WINDOW : len(test_data)]


    val_errors = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            mse = torch.mean((outputs - batch_y) ** 2, dim=1)
            val_errors.extend(mse.numpy())
    
    # Calculate and save threshold bounds
    threshold_high = 1.0*np.percentile(val_errors, 95)  # Upper threshold (95th percentile)
    threshold_low = np.percentile(val_errors, 30)    # Lower threshold (30th percentile)
    print (f"threshold_high:", threshold_high)
    # Save both thresholds to config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if "lstm" not in config:
        config["lstm"] = {}
    config["lstm"]["THRESHOLD_HIGH"] = float(threshold_high*1)
   
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nLSTM threshold settings:")
    print(f"Upper threshold (anomaly): {threshold_high:.4f}")
    print(f"Lower threshold (normal): {threshold_low:.4f}")

    test_labels = np.zeros(len(test_data) - TIME_WINDOW, dtype=int)
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
            outputs = model(batch_x)
            mse = torch.mean((outputs - batch_y) ** 2, dim=1)
            
            # Calculate batch position in final results
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + len(mse)

            test_labels[start_idx:end_idx] = (mse > threshold_high).int().numpy()
    
    print(f"True labels length: {len(true_labels)}")
    print(f"Predicted labels length: {len(test_labels)}")

    # Adjust true_labels length to match test_labels
    adjusted_true_labels = true_labels[:len(test_labels)]
    cm, accuracy, precision, recall, f1_score = compute_metrics(adjusted_true_labels, test_labels)
    # Print statistics
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
    #save_metrics_to_csv("LSTM", metrics)
    # Print confusion matrix
    print("\nConfusion matrix:")
    print("          Predicted")
    print("          Normal Anomaly")
    print(f"Actual Normal  {cm[0,0]:<6} {cm[0,1]:<6}")
    print(f"       Anomaly {cm[1,0]:<6} {cm[1,1]:<6}")
