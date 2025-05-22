#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于风险自适应注入噪声的重构攻击评估
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
# 全局配置
# ================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = "subset.csv"
OUT_DIR   = "dynamic_ci_evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

# 数据拆分参数
TRAIN_RATIO = 0.9
FEATURES    = ["temperature", "humidity"]
CONF_LEVEL  = 0.95
REPEATS     = 5    # 正式可改大次数

# 差分隐私常量
DELTA         = 1e-6
C             = np.sqrt(2 * np.log(1.25 / DELTA))
delta_f       = 1.0
SENSITIVITIES = {"temperature": 0.686, "humidity": 1.3696}

# 风险与自适应参数
R_risk  = np.arange(0.0, 1.01, 0.1)
k       = 0.2
E_ratio = 1.0
lam     = 0.5

# 基准噪声对应的 ε(r)
sigma_base_row = np.array([
    4.963, 4.884, 4.698, 4.403, 4.047,
    3.562, 3.266, 2.878, 2.519, 2.211, 1.967
])
epsilon_r = delta_f * C / sigma_base_row

# ================
# Autoencoder 定义
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
# 加载数据
# ================
df      = pd.read_csv(DATA_PATH)
n_total = len(df)
n_train = int(n_total * TRAIN_RATIO)
orig90  = df[FEATURES].iloc[:n_train].values.astype(float)
hold10  = df[FEATURES].iloc[n_train:].values.astype(float)

# 置信区间常数
z = stats.norm.ppf((1 + CONF_LEVEL) / 2)

all_results = []

# ================
# 主循环
# ================
for idx, r in enumerate(R_risk):
    eps = epsilon_r[idx]
    print(f"===== r={r:.1f}  ε={eps:.3f} =====")
    result_row = {"risk": r, "epsilon": eps}

    for fi, feat in enumerate(FEATURES):
        sens = SENSITIVITIES[feat]
        hit_rates = []

        for _ in range(REPEATS):
            # 1) 计算自适应 σ_noise
            sigma_base = sens * eps
            sigma_t = sigma_base * (k + E_ratio) if r < 0.5 else sigma_base * np.exp(lam * r)
            sigma_noise = max(sigma_t, sigma_base)

            # 2) 后10% ε=1 噪声
            sigma_hold = sens * 1
            hold_pert  = hold10[:, fi] + np.random.normal(0, sigma_hold, size=hold10.shape[0])

            # 3) 前90% r 下噪声
            train_pert = orig90[:, fi] + np.random.normal(0, sigma_noise, size=n_train)
            train_pert = train_pert.reshape(-1, 1)

            # 4) 基于 hold_pert 计算 CI
            mu_h    = hold_pert.mean()
            sd_h    = hold_pert.std(ddof=1)
            ci_low  = mu_h - z * sd_h
            ci_high = mu_h + z * sd_h

            # 5) 训练自编码器 & 重构
            model = train_autoencoder(train_pert, epochs=300)
            with torch.no_grad():
                inp   = torch.FloatTensor(train_pert).to(model.encoder[0].weight.device)
                recon = model(inp).cpu().numpy().flatten()

            # 6) 命中率
            hr = np.mean((recon >= ci_low) & (recon <= ci_high))
            hit_rates.append(hr)

        mean_hr = np.mean(hit_rates)
        sem_hr  = np.std(hit_rates, ddof=1) / np.sqrt(REPEATS)
        result_row[f"{feat}_hit_rate"] = mean_hr
        result_row[f"{feat}_hr_sem"]    = sem_hr
        print(f"  {feat}: {mean_hr:.2%} ± {sem_hr:.2%}")

    all_results.append(result_row)

# ================
# 保存结果
# ================
df_res = pd.DataFrame(all_results)
df_res.to_csv(os.path.join(OUT_DIR, "dynamic_ci_results.csv"), index=False)

# ================
# 可视化：仅放大字体
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

print("Done. 结果和图表已保存至", OUT_DIR)
