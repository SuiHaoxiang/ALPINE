import time
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
from pathlib import Path

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_data(data_path, features):
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features].values.astype(float))
    return scaled_features, df['label'].values

def main():
    config = load_config()
    test_data_path = config["shared"]["data"]["test_data"]
    df = pd.read_csv(test_data_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col != 'label']
    if not features:
        raise ValueError("没有找到数值特征列")
    model_path = "../svm_model_fd.pkl"
    
    model = load_model(model_path)
    X_test, y_true = load_data(test_data_path, features)
    
    start_time = time.time()
    scores = model.decision_function(X_test)
    predictions = model.predict(X_test)
    end_time = time.time()
    
    avg_time_per_sample = (end_time - start_time) / len(y_true)
    print(f"推理时间: {avg_time_per_sample*1000:.6f} ms/样本")

if __name__ == "__main__":
    main()
