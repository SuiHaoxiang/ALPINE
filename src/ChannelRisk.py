import pandas as pd
import numpy as np
import torch
import pickle
import yaml
from pathlib import Path
import sys
from lstm import TinyLSTM
from spa import SparseIsolationForest,IsolationTree,IsolationTreeNode
from sklearn.preprocessing import StandardScaler

class ChannelRisk:
    """Channel risk calculation class, encapsulates risk calculation logic"""

    def __init__(self, config_path="config.yaml"):
        """Initialize risk calculator

        Args:
            config_path (str): Path to config file
        """
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            hyb_config = config["hyb"]
            shared_config = config["shared"]
            lstm_config = config["lstm"]

        # Get configuration parameters
        self.TIME_WINDOW = shared_config["TIME_WINDOW"]
        self.HIDDEN_SIZE = lstm_config["HIDDEN_SIZE"]
        self.LSTM_TH = lstm_config["THRESHOLD_HIGH"]
        self.GLOBAL_HIGH_THRESHOLD = config["spa"]["GLOBAL_THRESHOLD"]
        self.GLOBAL_LOW_THRESHOLD = config["spa"]["NORMAL_THRESHOLD"]
        self.features = shared_config["features"]
        
        # Load test data for standardizer fitting
        test_df = pd.read_csv(shared_config["data"]["test_data"])
        
        # Initialize standardizer
        self.scaler = StandardScaler()
        self.scaler.fit(test_df[self.features].values.astype(float))
        
        # Load models
        with open(hyb_config["models"]["sif_model"], "rb") as f:
            self.sif_model = pickle.load(f)

        self.lstm_model = TinyLSTM(
            input_size=3, 
            hidden_size=self.HIDDEN_SIZE, 
            output_size=3
        )
        self.lstm_model.load_state_dict(
            torch.load(hyb_config["models"]["lstm_model"])
        )
        self.lstm_model.eval()
        
        # Initialize history buffer
        self.history_buffer = []
    
    def calculate_risk(self, current_features):
        """Calculate risk value for current features
        
        Args:
            current_features (np.array): Feature array for current timestep
            
        Returns:
            tuple: (risk value, decision reason)
        """
        # Standardize current features
        scaled_features = self.scaler.transform(
            current_features.reshape(1, -1).astype(float))
        sif_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        
        # Calculate Sparse-iForest score
        s_score = self.sif_model.anomaly_score(sif_tensor).item()
        
        # First level judgment
        if s_score > self.GLOBAL_HIGH_THRESHOLD:
            return 1.0, "SIF-Anomaly"
        elif s_score < self.GLOBAL_LOW_THRESHOLD:
            return 0.1, "SIF-Normal"
        
        # Second level LSTM judgment
        if len(self.history_buffer) >= self.TIME_WINDOW:
            # Get time window data
            window_data = np.array(self.history_buffer[-self.TIME_WINDOW:])
            scaled_window = self.scaler.transform(window_data.astype(float))
            lstm_input = torch.FloatTensor(scaled_window).unsqueeze(0)
            
            # LSTM prediction
            with torch.no_grad():
                pred = self.lstm_model(lstm_input)
                true = torch.FloatTensor(scaled_features)
                delta = torch.mean((pred - true) ** 2).item()
            
            if delta > self.LSTM_TH:
                return 1.0, "LSTM-Anomaly"
            else:
                return 0.1, "LSTM-Judged"
        else:
            return 1, "Insufficient-History"
    
    def update_history(self, current_features):
        """Update history buffer
        
        Args:
            current_features (np.array): Feature array for current timestep
        """
        self.history_buffer.append(current_features)
    
    def process_sample(self, features):
        """Process single sample
        
        Args:
            features (np.array): Feature array for current sample
            
        Returns:
            tuple: (risk value, decision reason)
        """
        self.update_history(features)
        return self.calculate_risk(features)

