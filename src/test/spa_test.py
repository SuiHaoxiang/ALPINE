import time
import pickle
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from spa import SparseIsolationForest, IsolationTree, IsolationTreeNode

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_data(data_path, features):
    data = pd.read_csv(data_path)
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return torch.tensor(data[features].values, dtype=torch.float32), data['label'].values

def main():
    model_path = "../sparse_isolation_forest_model_ucr.pkl"
    test_data_path = "../../dataset/UCR/csv_files/136/test.csv"
    features = ["value"]
    
    model = load_model(model_path)
    X_test, _ = load_data(test_data_path, features)
    
    start_time = time.time()
    scores = model.anomaly_score(X_test)
    end_time = time.time()
    
    avg_time_per_sample = (end_time - start_time) / len(X_test)
    print(f"推理时间: {avg_time_per_sample*1000:.6f} ms/样本")

if __name__ == "__main__":
    main()
