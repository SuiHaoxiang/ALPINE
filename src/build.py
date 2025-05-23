import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import psutil
import torch
import torch.nn as nn
class Config:
    STATE_DIM    = 1     # State dimension (risk value rrisk)
    ACTION_DIM   = 1     # Action dimension (privacy budget epsilon)
    EPSILON_MIN  = 1.0   # Minimum privacy budget
    EPSILON_MAX  = 5.0   # Maximum privacy budget
    ALPHA        = 5     # Privacy gain weight
    BETA         = 20    # Utility loss weight
    GAMMA        = 0.99  # Discount factor
    TAU          = 0.005 # Soft update coefficient
    ACTOR_LR     = 1e-4  # Actor learning rate
    CRITIC_LR    = 1e-3  # Critic learning rate
    BUFFER_SIZE  = 100000
    BATCH_SIZE   = 64
    OU_THETA     = 0.2   # OU noise parameter
    OU_SIGMA     = 0.5   # OU noise intensity

    # Privacy gain hyperparameters
    PRIV_KAPPA   = 5.0  # Logistic steepness
    PRIV_S0      = 0.8   # Logistic center point
    PRIV_DELTA   = 0.7   # Budget power penalty

    # Utility loss hyperparameters
    UTIL_RHO     = 0.5   # Risk coupling coefficient
    UTIL_SIGMA0  = 1.0   # Noise baseline standard deviation

    # Nonlinear smooth state transition hyperparameters
    TRANS_ETA    = 0.2   # Smoothing step size
    TRANS_GAMMA  = 2.0   # Power term


# ---------------------- Load Actor Model ----------------------
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.STATE_DIM, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, Config.ACTION_DIM),
            nn.Tanh()  # Output range [-1,1]
        )
    
    def forward(self, state):
        x = self.net(state)
        eps = (x + 1.0) * 0.5 * (Config.EPSILON_MAX - Config.EPSILON_MIN) + Config.EPSILON_MIN
        return eps.clamp(Config.EPSILON_MIN, Config.EPSILON_MAX)

actor = Actor()
actor.load_state_dict(torch.load("actor_model.pth"))
actor.eval()

# ---------------------- Global Configuration ----------------------
DELTA = 1e-6
C = np.sqrt(2 * np.log(1.25 / DELTA))
delta_f = 1.0

k = 0.2
E_ratio = 1.0
lam = 0.5

R_risk = np.arange(0.0, 1.01, 0.1)

DATA_PATH   = "subset.csv"
TRAIN_RATIO = 0.9
OUTPUT_DIR  = "dp_dynamic_with_actor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = ["temperature", "humidity"]
SENSITIVITIES = {
    "temperature": 0.686,
    "humidity":    1.3696
}

# ---------------------- Loading and Preprocessing ----------------------
df = pd.read_csv(DATA_PATH)
n90 = int(len(df) * TRAIN_RATIO)
orig90 = df[FEATURES].iloc[:n90].copy()
test10 = df[FEATURES].iloc[n90:].copy()

orig_means = orig90.mean()
orig_stds  = orig90.std(ddof=1)
test_stds  = test10.std(ddof=1)

# ---------------------- Result Containers ----------------------
records_sigma   = []
records_wass    = []
records_anomaly = []
records_mem     = []

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss

# ---------------------- Main Loop ----------------------
for r in R_risk:
    # Get epsilon from Actor model
    with torch.no_grad():
        eps = actor(torch.tensor([[r]], dtype=torch.float32)).item()

    row_sigma   = {"risk": r, "epsilon": eps}
    row_wass    = {"risk": r, "epsilon": eps}
    row_anomaly = {"risk": r, "epsilon": eps}

    noised = orig90.copy()

    for feat in FEATURES:
        Δ = SENSITIVITIES[feat]
        sigma_base = Δ * eps

        if r < 0.5:
            sigma_t = sigma_base * (k + E_ratio)
        else:
            sigma_t = sigma_base * np.exp(lam * r)
        sigma_noise = max(sigma_t, sigma_base)

        row_sigma[feat] = sigma_noise
        noise = np.random.normal(0, sigma_noise, size=n90)
        noised_feat = noised[feat] + noise

        w = wasserstein_distance(test10[feat], noised_feat)
        row_wass[feat] = w / test_stds[feat]

        lo = orig_means[feat] - 3 * orig_stds[feat]
        hi = orig_means[feat] + 3 * orig_stds[feat]
        row_anomaly[feat] = np.mean((noised_feat < lo) | (noised_feat > hi))

    # Record memory usage
    mem_after = process.memory_info().rss
    row_mem = {"risk": r, "rss_bytes": mem_after - mem_before}
    records_mem.append(row_mem)

    records_sigma.append(row_sigma)
    records_wass.append(row_wass)
    records_anomaly.append(row_anomaly)

# ---------------------- Convert to DataFrame ----------------------
df_sigma   = pd.DataFrame(records_sigma)
df_wass    = pd.DataFrame(records_wass)
df_anomaly = pd.DataFrame(records_anomaly)
df_mem     = pd.DataFrame(records_mem)

# ---------------------- Visualization Functions ----------------------
def plot_metric(df, y_label, fname, ylim=None):
    plt.figure(figsize=(8,5))
    for feat in FEATURES:
        plt.plot(df["risk"], df[feat],
                 marker='o', linewidth=2, markersize=6,
                 label=feat.capitalize())
    plt.xlabel("Risk Level (r)", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(R_risk)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
    plt.close()

plot_metric(df_sigma,   "Noise Std Dev σ_noise",   "noise_sigma_vs_risk.png")
plot_metric(df_wass,    "Wasserstein/Std",         "wass_vs_risk.png",    ylim=(1.5,3.5))
plot_metric(df_anomaly, "Anomaly Ratio",           "anomaly_vs_risk.png", ylim=(0,0.2))

# ---------------------- Save CSV Files ----------------------
df_sigma.to_csv(os.path.join(OUTPUT_DIR, "sigma_dp.csv"), index=False)
df_wass.to_csv(os.path.join(OUTPUT_DIR, "wass_dp.csv"), index=False)
df_anomaly.to_csv(os.path.join(OUTPUT_DIR, "anomaly_dp.csv"), index=False)
df_mem.to_csv(os.path.join(OUTPUT_DIR, "memory_usage.csv"), index=False)

# ---------------------- Print Memory Info ----------------------
print("Memory usage delta (RSS) per risk level saved to memory_usage.csv")
