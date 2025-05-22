import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import psutil
import torch
import torch.nn as nn
class Config:
    STATE_DIM    = 1     # 状态维度 (风险值 rrisk)
    ACTION_DIM   = 1     # 动作维度 (隐私预算 epsilon)
    EPSILON_MIN  = 1.0   # 最小隐私预算
    EPSILON_MAX  = 5.0   # 最大隐私预算
    ALPHA        = 5 # 隐私增益权重
    BETA         = 20  # 效用损失权重
    GAMMA        = 0.99  # 折扣因子
    TAU          = 0.005 # 软更新系数
    ACTOR_LR     = 1e-4  # Actor 学习率
    CRITIC_LR    = 1e-3  # Critic 学习率
    BUFFER_SIZE  = 100000
    BATCH_SIZE   = 64
    OU_THETA     = 0.2   # OU 噪声参数
    OU_SIGMA     = 0.5   # OU 噪声强度

    # 隐私增益超参数
    PRIV_KAPPA   = 5.0  # Logistic 陡峭度
    PRIV_S0      = 0.8   # Logistic 中心点
    PRIV_DELTA   = 0.7   # 预算幂次惩罚

    # 效用损失超参数
    UTIL_RHO     = 0.5   # 风险耦合系数
    UTIL_SIGMA0  = 1.0   # 噪声基准标准差

    # 非线性平滑状态转移超参数
    TRANS_ETA    = 0.2   # 平滑步长
    TRANS_GAMMA  = 2.0   # 幂次


# ---------------------- 加载Actor模型 ----------------------
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
            nn.Tanh()  # 输出范围[-1,1]
        )
    
    def forward(self, state):
        x = self.net(state)
        eps = (x + 1.0) * 0.5 * (Config.EPSILON_MAX - Config.EPSILON_MIN) + Config.EPSILON_MIN
        return eps.clamp(Config.EPSILON_MIN, Config.EPSILON_MAX)

actor = Actor()
actor.load_state_dict(torch.load("actor_model.pth"))
actor.eval()

# ---------------------- 全局配置 ----------------------
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

# ---------------------- 加载与预处理 ----------------------
df = pd.read_csv(DATA_PATH)
n90 = int(len(df) * TRAIN_RATIO)
orig90 = df[FEATURES].iloc[:n90].copy()
test10 = df[FEATURES].iloc[n90:].copy()

orig_means = orig90.mean()
orig_stds  = orig90.std(ddof=1)
test_stds  = test10.std(ddof=1)

# ---------------------- 结果容器 ----------------------
records_sigma   = []
records_wass    = []
records_anomaly = []
records_mem     = []

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss

# ---------------------- 主循环 ----------------------
for r in R_risk:
    # 使用Actor模型获得 epsilon
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

    # 记录内存占用
    mem_after = process.memory_info().rss
    row_mem = {"risk": r, "rss_bytes": mem_after - mem_before}
    records_mem.append(row_mem)

    records_sigma.append(row_sigma)
    records_wass.append(row_wass)
    records_anomaly.append(row_anomaly)

# ---------------------- 转为 DataFrame ----------------------
df_sigma   = pd.DataFrame(records_sigma)
df_wass    = pd.DataFrame(records_wass)
df_anomaly = pd.DataFrame(records_anomaly)
df_mem     = pd.DataFrame(records_mem)

# ---------------------- 可视化函数 ----------------------
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

# ---------------------- 保存 CSV ----------------------
df_sigma.to_csv(os.path.join(OUTPUT_DIR, "sigma_dp.csv"), index=False)
df_wass.to_csv(os.path.join(OUTPUT_DIR, "wass_dp.csv"), index=False)
df_anomaly.to_csv(os.path.join(OUTPUT_DIR, "anomaly_dp.csv"), index=False)
df_mem.to_csv(os.path.join(OUTPUT_DIR, "memory_usage.csv"), index=False)

# ---------------------- 打印内存信息 ----------------------
print("Memory usage delta (RSS) per risk level saved to memory_usage.csv")

