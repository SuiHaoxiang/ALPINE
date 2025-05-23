#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk-adaptive noise injection for reconstruction attack evaluation
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
import matplotlib.pyplot as plt

# ================
# Global configuration
# ================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = "subset.csv"
OUT_DIR   = "dynamic_ci_evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

# Data splitting parameters
TRAIN_RATIO = 0.9
FEATURES    = ["temperature", "humidity"]
CONF_LEVEL  = 0.95
REPEATS     = 5   

# Differential privacy constants
DELTA         = 1e-6
C             = np.sqrt(2 * np.log(1.25 / DELTA))
delta_f       = 1.0
SENSITIVITIES = {"temperature": 0.686, "humidity": 1.3696}

# Risk and adaptive parameters
R_risk  = np.arange(0.0, 1.01, 0.1)
k       = 0.2
E_ratio = 1.0
lam     = 0.5

# Base noise for ε(r)
sigma_base_row = np.array([
    4.963, 4.884, 4.698, 4.403, 4.047,
    3.562, 3.266, 2.878, 2.519, 2.211, 1.967
])
epsilon_r = delta_f * C / sigma_base_row

# ================
# Autoencoder definition
# ================
class Autoencoder1D(nn.Module):
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

def train_autoencoder(data_np, epochs=200, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Autoencoder1D().to(device)
    opt    = optim.Adam(model.parameters(), lr=lr)
    crit   = nn.MSELoss()
    X      = torch.FloatTensor(data_np).to(device)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out  = model(X)
        loss = crit(out, X)
        loss.backward()
        opt.step()
    return model

# ================
# Load data
# ================
df      = pd.read_csv(DATA_PATH)
n_total = len(df)
n_train = int(n_total * TRAIN_RATIO)
orig90  = df[FEATURES].iloc[:n_train].values.astype(float)
hold10  = df[FEATURES].iloc[n_train:].values.astype(float)

# Confidence interval constant
z = stats.norm.ppf((1 + CONF_LEVEL) / 2)

all_results = []

# ================
# Main loop
# ================
for idx, r in enumerate(R_risk):
    eps = epsilon_r[idx]
    print(f"===== r={r:.1f}  ε={eps:.3f} =====")
    result_row = {"risk": r, "epsilon": eps}

    for fi, feat in enumerate(FEATURES):
        sens = SENSITIVITIES[feat]
        hit_rates = []

        for _ in range(REPEATS):
            # 1) Calculate adaptive σ_noise
            sigma_base = sens * eps
            sigma_t = sigma_base * (k + E_ratio) if r < 0.5 else sigma_base * np.exp(lam * r)
            sigma_noise = max(sigma_t, sigma_base)

            # 2) Add ε=1 noise to last 10%
            sigma_hold = sens * 1
            hold_pert  = hold10[:, fi] + np.random.normal(0, sigma_hold, size=hold10.shape[0])

            # 3) Add noise to first 90% based on r
            train_pert = orig90[:, fi] + np.random.normal(0, sigma_noise, size=n_train)
            train_pert = train_pert.reshape(-1, 1)

            # 4) Calculate CI from hold_pert
            mu_h    = hold_pert.mean()
            sd_h    = hold_pert.std(ddof=1)
            ci_low  = mu_h - z * sd_h
            ci_high = mu_h + z * sd_h

            # 5) Train autoencoder & reconstruct
            model = train_autoencoder(train_pert, epochs=300)
            with torch.no_grad():
                inp   = torch.FloatTensor(train_pert).to(model.encoder[0].weight.device)
                recon = model(inp).cpu().numpy().flatten()

            # 6) Calculate hit rate
            hr = np.mean((recon >= ci_low) & (recon <= ci_high))
            hit_rates.append(hr)

        mean_hr = np.mean(hit_rates)
        sem_hr  = np.std(hit_rates, ddof=1) / np.sqrt(REPEATS)
        result_row[f"{feat}_hit_rate"] = mean_hr
        result_row[f"{feat}_hr_sem"]    = sem_hr
        print(f"  {feat}: {mean_hr:.2%} ± {sem_hr:.2%}")

    all_results.append(result_row)

# ================
# Save results
# ================
df_res = pd.DataFrame(all_results)
df_res.to_csv(os.path.join(OUT_DIR, "dynamic_ci_results.csv"), index=False)

# ================
# Visualization: only enlarge fonts
# ================
plt.rcParams['font.size']       = 14
plt.rcParams['axes.labelsize']  = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['grid.color']      = '#666666'
plt.rcParams['grid.linestyle']  = '--'
plt.rcParams['grid.linewidth']  = 0.6

plt.figure(figsize=(8, 5))
for feat in FEATURES:
    plt.plot(
        df_res["risk"],
        df_res[f"{feat}_hit_rate"] * 100,
        marker='o',
        markersize=7,
        label=feat.capitalize()
    )

plt.xlabel("Risk Level")
plt.ylabel("Hit Rate (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "69_large_font.png"), dpi=300)
plt.close()

print("Done. Results and plots saved to", OUT_DIR)
