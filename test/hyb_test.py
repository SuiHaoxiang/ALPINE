import time
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from lstm import TinyLSTM, TimeSeriesDataset
from spa import SparseIsolationForest,IsolationTree,IsolationTreeNode
import pickle

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def load_models(config):
    lstm_model = TinyLSTM(
        input_size=3, 
        hidden_size=config["lstm"]["HIDDEN_SIZE"],
        output_size=3
    )
    lstm_model.load_state_dict(torch.load(config["hyb"]["models"]["lstm_model"]))
    lstm_model.eval()
    
    with open(config["hyb"]["models"]["sif_model"], "rb") as f:
        sif_model = pickle.load(f)
    
    return lstm_model, sif_model

def load_data(data_path, features, time_window):
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features].values.astype(float))
    return scaled_features, df['label'].values, scaler

def main():
    config = load_config()
    time_window = config["shared"]["TIME_WINDOW"]
    features = config["shared"]["features"]
    test_data_path = config["shared"]["data"]["test_data"]
    
    lstm_model, sif_model = load_models(config)
    test_features, true_labels, scaler = load_data(test_data_path, features, time_window)
    
    lstm_th = config["lstm"]["THRESHOLD_HIGH"]
    global_high_th = config["spa"]["GLOBAL_THRESHOLD"]
    global_low_th = config["spa"]["NORMAL_THRESHOLD"]
    
    test_results = {'per_sample_times': []}
    
    for _ in range(5):
        final_pred = np.zeros(len(true_labels), dtype=int)
        start_time = time.time()
        
        for i in range(time_window, len(test_features)):
            current_features = test_features[i].reshape(1, -1)
            sif_tensor = torch.tensor(current_features, dtype=torch.float32)
            global_score = sif_model.anomaly_score(sif_tensor).item()
            
            if global_score > global_high_th:
                final_pred[i] = 1
                continue
            elif global_score < global_low_th:
                final_pred[i] = 0 
                continue
                
            window_data = test_features[i-time_window:i]
            lstm_input = torch.FloatTensor(window_data).unsqueeze(0)
            
            with torch.no_grad():
                pred = lstm_model(lstm_input)
                true = torch.FloatTensor(current_features)
                mse = torch.mean((pred - true) ** 2).item()
                
            final_pred[i] = 1 if mse > lstm_th else 0
        
        end_time = time.time()
        test_results['per_sample_times'].append(
            (end_time - start_time) / (len(test_features) - time_window))
    
    avg_time = np.mean(test_results['per_sample_times']) * 1000
    # 注意：混合模型包含Sparse-iForest和LSTM两阶段判断，时间会比其他单一模型长
    print(f"推理时间: {avg_time:.6f} ms/样本 (混合模型)")

if __name__ == "__main__":
    main()
