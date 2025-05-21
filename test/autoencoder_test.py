import time
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
from pathlib import Path

class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, encoding_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim*2, encoding_dim))
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, encoding_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim*2, input_dim))
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(config):
    input_dim = len(config["shared"]["features"])
    model = Autoencoder(input_dim, config["autoencoder"]["ae_encoding_dim"])
    model.load_state_dict(torch.load(config["autoencoder"]["model_path"]))
    model.eval()
    return model

def load_data(data_path, features):
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features].values.astype(float))
    return torch.FloatTensor(scaled_features), df['label'].values

def main():
    config = load_config()
    test_data_path = config["shared"]["data"]["test_data"]
    features = config["shared"]["features"]
    threshold_quantile = config["autoencoder"]["ae_threshold_quantile"]
    
    model = load_model(config)
    X_test, y_true = load_data(test_data_path, features)
    
    test_results = {'per_sample_times': []}
    
    for _ in range(5):
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(X_test)
            mse = torch.mean((outputs - X_test)**2, dim=1)
        
        end_time = time.time()
        test_results['per_sample_times'].append(
            (end_time - start_time) / len(y_true))
    
    avg_time = np.mean(test_results['per_sample_times']) * 1000
    print(f"推理时间: {avg_time:.6f} ms/样本")

if __name__ == "__main__":
    main()
