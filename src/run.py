#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated evaluation script - Combines autop and build functionality, primarily using build's evaluation methods
"""

import os
import numpy as np
import pandas as pd
import torch
import yaml
from Risk import RiskCalculator
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import psutil
from lstm import TinyLSTM
from spa import SparseIsolationForest,IsolationTree,IsolationTreeNode
from ChannelRisk import ChannelRisk
from SemanticRisk import SemanticRiskCalculator
with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        shared_config = config["shared"]
# ==================== Configuration Parameters ====================
class Config:
    # Data parameters
    DATA_PATH = "../dataset/SensorData/subset.csv"
    OUTPUT_DIR = "combined_evaluation"
    FEATURES = ["temperature", "humidity"]
    SENSITIVITIES = {"temperature": 0.686, "humidity": 1.3696}
    TRAIN_RATIO = 0.9
    
    # Privacy parameters
    DELTA = 1e-6
    C = np.sqrt(2 * np.log(1.25 / DELTA))
    DELTA_F = 1.0
    
    # Adaptive noise parameters
    K = 0.2
    E_RATIO = 1.0
    LAM = 0.5
    
    # Evaluation parameters
    CONF_LEVEL = 0.95
    REPEATS = 5
    
    
    OU_THETA = 0.2   # OU noise parameter
    OU_SIGMA = 0.5   # OU noise intensity
    PRIV_KAPPA = 5.0 # Logistic steepness
    PRIV_S0 = 0.8    # Logistic center point
    PRIV_DELTA = 0.7 # Budget power penalty
    UTIL_RHO = 0.5   # Risk coupling coefficient
    UTIL_SIGMA0 = 1.0 # Base noise standard deviation
    TRANS_ETA = 0.2  # Smoothing step size
    TRANS_GAMMA = 2.0 # Power
    ACTOR_LR = 1e-4  # Actor learning rate
    CRITIC_LR = 1e-3 # Critic learning rate
    BUFFER_SIZE = 100000 # Experience replay buffer size
    BATCH_SIZE = 64  # Batch size
    # Privacy gain hyperparameters
    PRIV_KAPPA   = 5.0  # Logistic steepness
    PRIV_S0      = 0.8   # Logistic center point
    PRIV_DELTA   = 0.7   # Budget power penalty

    # Utility loss hyperparameters
    UTIL_RHO     = 0.5   # Risk coupling coefficient
    UTIL_SIGMA0  = 1.0   # Base noise standard deviation

    # Nonlinear smooth state transition hyperparameters
    TRANS_ETA    = 0.2   # Smoothing step size
    TRANS_GAMMA  = 2.0   # Power

# ==================== Autoencoder Model ====================
class Autoencoder1D(nn.Module):
    """Autoencoder model inherited from autop.py"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(data, epochs=200, lr=1e-3):
    """Train autoencoder"""
    model = Autoencoder1D()  # Force using CPU
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    data_tensor = torch.FloatTensor(data.reshape(-1, 1))
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(data_tensor)
        loss = criterion(outputs, data_tensor)
        loss.backward()
        optimizer.step()
    
    return model

# ==================== Main Evaluator Class ====================
class PrivacyEvaluator:
    def __init__(self, actor_model_path=None):
        """Initialize evaluator"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        self.actor = self._load_actor(actor_model_path) if actor_model_path else None
        self.z_score = stats.norm.ppf((1 + Config.CONF_LEVEL) / 2)
        
    def _load_actor(self, model_path):
        """Load pretrained Actor model"""
        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 128), nn.LayerNorm(128), nn.ELU(),
                    nn.Linear(128, 64), nn.LayerNorm(64), nn.ELU(),
                    nn.Linear(64, 1), nn.Tanh()
                )
            
            def forward(self, state):
                x = self.net(state)
                return (x + 1) * 2 + 1  # Output range [1,5]
        
        actor = Actor()
        if model_path and os.path.exists(model_path):
            actor.load_state_dict(torch.load(model_path))
        return actor
    
    def evaluate(self):
        """Execute complete evaluation process"""
        print("="*50)
        print("Starting evaluation process...")
        print(f"Data file: {Config.DATA_PATH}")
        print(f"Output directory: {Config.OUTPUT_DIR}")
        print(f"Features: {', '.join(Config.FEATURES)}")
        print("Risk levels: dynamic calculation based on combined risk")
        print("="*50)
        
        # Load data
        df = pd.read_csv(Config.DATA_PATH)
        print(f"Successfully loaded data, total samples: {len(df)}")
        n_train = int(len(df) * Config.TRAIN_RATIO)
        train_data = df[Config.FEATURES].iloc[:n_train].values
        test_data = df[Config.FEATURES].iloc[n_train:].values
        
        # Prepare result container
        results = []
        process = psutil.Process(os.getpid())
        base_mem = process.memory_info().rss
        
        # Initialize risk calculators
        risk_calculator = RiskCalculator()
        channel_risk_calculator = ChannelRisk()
        calculator = SemanticRiskCalculator(w1=0.5, w2=0.5, window_size=5)
    
        channel_test_df = pd.read_csv("../dataset/FD/test.csv")
        semantic_df = pd.read_csv('../dataset/SensorData/subset.csv')
        use_columns = df.columns[:2]
        #r_channel_values = channel_risk_calculator.batch_process(channel_test_df)
        r_semantic_values = calculator.calculate(semantic_df, use_columns)

        
        # Calculate combined risk and evaluate
        for i in range(len(test_data)):
            features = channel_test_df[shared_config["features"]].iloc[i].values
            r_channel_values, _ = channel_risk_calculator.process_sample(features)
            r = risk_calculator.calculate_combined_risk(
                r_channel_values, r_semantic_values[i]['semantic_risk'])
            epsilon = self._get_epsilon(r)
            record = {"risk": r, "epsilon": epsilon}
            
            # Print risk level header
            print(f"\n===== r={r:.1f} ε={epsilon:.3f} =====")
            
            # Evaluate each feature
            for i, feat in enumerate(Config.FEATURES):
                # Calculate noise parameters
                sigma_base = Config.SENSITIVITIES[feat] * epsilon
                if r < 0.5:
                    sigma_t = sigma_base * (Config.K + Config.E_RATIO)
                else:
                    sigma_t = sigma_base * np.exp(Config.LAM * r)
                sigma_noise = max(sigma_t, sigma_base)
                
                # Add noise
                train_noised = train_data[:, i] + np.random.normal(0, sigma_noise, size=n_train)
                test_noised = test_data[:, i] + np.random.normal(0, sigma_base, size=len(test_data))
                
                # Calculate metrics
                wass_dist = wasserstein_distance(test_data[:, i], test_noised)
                record[f"{feat}_wass"] = wass_dist / test_data[:, i].std()
                
                mean, std = train_data[:, i].mean(), train_data[:, i].std()
                lo, hi = mean - 3*std, mean + 3*std
                record[f"{feat}_anomaly"] = np.mean((train_noised < lo) | (train_noised > hi))
                
                
                # Autoencoder evaluation
                hit_rates = []
                for _ in range(Config.REPEATS):
                    model = train_autoencoder(train_noised, epochs=300)
                    with torch.no_grad():
                        recon = model(torch.FloatTensor(train_noised.reshape(-1, 1))).numpy().flatten()
                    
                    # Calculate confidence interval
                    test_mean, test_std = test_noised.mean(), test_noised.std(ddof=1)
                    ci_low = test_mean - self.z_score * test_std
                    ci_high = test_mean + self.z_score * test_std
                    
                    hit_rate = np.mean((recon >= ci_low) & (recon <= ci_high))
                    hit_rates.append(hit_rate)
                
                record[f"{feat}_hit_rate"] = np.mean(hit_rates)
                record[f"{feat}_hit_sem"] = np.std(hit_rates, ddof=1) / np.sqrt(Config.REPEATS)
                
                # Print feature results in requested format
                print(f"  {feat}: {record[f'{feat}_hit_rate']*100:.2f}% ± {record[f'{feat}_hit_sem']*100:.2f}%")
            
            # Record memory usage
            record["memory_usage"] = process.memory_info().rss - base_mem
            results.append(record)
        
        # Save results
        self._save_results(results)
        print("\n" + "="*50)
        print("Evaluation completed! Results saved to", Config.OUTPUT_DIR)
        print(f"- Evaluation report: {os.path.join(Config.OUTPUT_DIR, 'evaluation_results.csv')}")
        print(f"- Visualization plots: {os.path.join(Config.OUTPUT_DIR, 'evaluation_plots.png')}")
        print("="*50)
    
    def _get_epsilon(self, r_risk):
        """Get privacy budget"""
        if self.actor:
            with torch.no_grad():
                return self.actor(torch.FloatTensor([[r_risk]])).item()
        return 5 - 4 * r_risk  # Linear fallback
    
    def _save_results(self, results):
        """Save evaluation results"""
        df = pd.DataFrame(results)
        
        # Save CSV
        df.to_csv(os.path.join(Config.OUTPUT_DIR, "evaluation_results.csv"), index=False)
        
        # Generate visualization plots
        self._generate_plots(df)
    
    def _generate_plots(self, df):
        """Generate visualization plots"""
        plt.figure(figsize=(15, 10))
        
        # Wasserstein distance
        plt.subplot(2, 2, 1)
        for feat in Config.FEATURES:
            plt.plot(df["risk"], df[f"{feat}_wass"], label=feat.capitalize())
        plt.xlabel("Risk Level")
        plt.ylabel("Wasserstein Distance")
        plt.title("Wasserstein Distance vs Risk")
        plt.legend()
        plt.grid(True)
        
        # Hit rate
        plt.subplot(2, 2, 2)
        for feat in Config.FEATURES:
            plt.errorbar(df["risk"], df[f"{feat}_hit_rate"], 
                        yerr=df[f"{feat}_hit_sem"], label=feat.capitalize())
        plt.xlabel("Risk Level")
        plt.ylabel("Hit Rate")
        plt.title("Reconstruction Hit Rate vs Risk")
        plt.legend()
        plt.grid(True)
        
        # Anomaly ratio
        plt.subplot(2, 2, 3)
        for feat in Config.FEATURES:
            plt.plot(df["risk"], df[f"{feat}_anomaly"], label=feat.capitalize())
        plt.xlabel("Risk Level")
        plt.ylabel("Anomaly Ratio")
        plt.title("Anomaly Ratio vs Risk")
        plt.legend()
        plt.grid(True)
        
        # Memory usage
        plt.subplot(2, 2, 4)
        plt.plot(df["risk"], df["memory_usage"] / (1024**2), 'k-')
        plt.xlabel("Risk Level")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage vs Risk")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "evaluation_plots.png"), dpi=300)
        plt.close()

# ==================== Main Program ====================
if __name__ == "__main__":
    evaluator = PrivacyEvaluator(actor_model_path="actor_model.pth")
    evaluator.evaluate()
