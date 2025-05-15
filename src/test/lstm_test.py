import time
import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from lstm import TinyLSTM, TimeSeriesDataset

def load_model(model_path, input_size=3, hidden_size=4, output_size=3):
    model = TinyLSTM(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_data(data_path, features, time_window):
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features].values.astype(float))
    dataset = TimeSeriesDataset(scaled_features, time_window)
    return DataLoader(dataset, batch_size=32, shuffle=False), df['label'].values[time_window:]

def main():
    model_path = "../best_model_fd.pth"
    test_data_path = "../../dataset/data_FD/test.csv"
    features = ["RSSI","Link Quality","Ping Delay"]
    # model_path = "../best_model_sd.pth"
    # test_data_path = "../../dataset/data_SD/test.csv"
    # features = ["signal_strength", "transmit_rate", "ping_latency"]
    time_window = 10
    
    model = load_model(model_path)
    test_loader, y_true = load_data(test_data_path, features, time_window)
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            outputs = model(batch_x)
    
    end_time = time.time()
    
    avg_time_per_sample = (end_time - start_time) / len(y_true)
    print(f"推理时间: {avg_time_per_sample*1000:.6f} ms/样本")

if __name__ == "__main__":
    main()
