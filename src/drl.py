import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------
# 超参数配置
# ----------------------

font_path = 'calibri.ttf'
font_manager.fontManager.addfont(font_path)

plt.rcParams['font.family']         = 'sans-serif'
plt.rcParams['font.sans-serif']     = ['Calibri']
plt.rcParams['font.weight']         = 'bold'        # 所有文字加粗
plt.rcParams['axes.titleweight']    = 'bold'        # 坐标轴标题加粗
plt.rcParams['axes.labelweight']    = 'bold'        # 坐标轴标签加粗
plt.rcParams['text.color']          = 'black'       # 确保文字够深
plt.rcParams['xtick.color']         = 'black'
plt.rcParams['ytick.color']         = 'black'
plt.rcParams['lines.linewidth']     = 2.0           # 折线更粗些，提高对比度
plt.rcParams['grid.color']          = '#666666'     # 网格也深一点
plt.rcParams['grid.linestyle']      = '--'
plt.rcParams['grid.linewidth']      = 0.5

class Config:
    STATE_DIM    = 1     # 状态维度 (风险值 rrisk)
    ACTION_DIM   = 1     # 动作维度 (隐私预算 epsilon)
    EPSILON_MIN  = 1.0   # 最小隐私预算
    EPSILON_MAX  = 5.0   # 最大隐私预算
    ALPHA        = 5     # 隐私增益权重
    BETA         = 20    # 效用损失权重
    GAMMA        = 0.99  # 折扣因子
    TAU          = 0.005 # 软更新系数
    ACTOR_LR     = 1e-4  # Actor 学习率
    CRITIC_LR    = 1e-3  # Critic 学习率
    BUFFER_SIZE  = 100000
    BATCH_SIZE   = 64
    OU_THETA     = 0.2   # OU 噪声参数
    OU_SIGMA     = 0.5   # OU 噪声强度

    # 隐私增益超参数
    PRIV_KAPPA   = 5.0   # Logistic 陡峭度
    PRIV_S0      = 0.7   # Logistic 中心点
    PRIV_DELTA   = 0.7   # 预算幂次惩罚

    # 效用损失超参数
    UTIL_RHO     = 0.5   # 风险耦合系数
    UTIL_SIGMA0  = 1.0   # 噪声基准标准差

    # 非线性平滑状态转移超参数
    TRANS_ETA    = 0.2   # 平滑步长
    TRANS_GAMMA  = 2.0   # 幂次


# ----------------------
# Actor网络 (输出1–5的隐私预算)
# ----------------------
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


# ----------------------
# Critic网络 (评估Q值)
# ----------------------
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.STATE_DIM + Config.ACTION_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


# ----------------------
# 改进的OU噪声生成器
# ----------------------
class OUNoise:
    def __init__(self):
        self.theta = Config.OU_THETA
        self.sigma = Config.OU_SIGMA
        self.mu = 0.0
        self.reset()

    def reset(self):
        self.state = np.ones(Config.ACTION_DIM) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(Config.ACTION_DIM)
        self.state += dx
        return self.state * 0.5  # 缩放


# ----------------------
# Replay Buffer
# ----------------------
class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.capacity = Config.BUFFER_SIZE
        self.pos = 0

    def add(self, s, a, r, s_next):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        s      = np.asarray(s, dtype=np.float32).reshape(-1)
        a      = np.asarray(a, dtype=np.float32).reshape(-1)
        s_next = np.asarray(s_next, dtype=np.float32).reshape(-1)
        self.buffer[self.pos] = (s, a, float(r), s_next)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self):
        idx   = np.random.choice(len(self.buffer), Config.BATCH_SIZE, replace=False)
        batch = [self.buffer[i] for i in idx]
        states      = np.vstack([x[0] for x in batch]).astype(np.float32)
        actions     = np.vstack([x[1] for x in batch]).astype(np.float32)
        rewards     = np.array([x[2] for x in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.vstack([x[3] for x in batch]).astype(np.float32)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states)
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------
# DDPG 智能体 (改进版)
# ----------------------
class DDPGAgent:
    def __init__(self):
        self.actor         = Actor()
        self.actor_target  = Actor()
        self.critic        = Critic()
        self.critic_target = Critic()

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim  = optim.Adam(self.actor.parameters(), lr=Config.ACTOR_LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=Config.CRITIC_LR)

        self.buffer = ReplayBuffer()
        self.noise  = OUNoise()

    def select_action(self, state, exploration=True):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            eps = self.actor(state_t).cpu().numpy().flatten()[0]
        if exploration:
            eps += float(self.noise.sample()[0])
        return float(np.clip(eps, Config.EPSILON_MIN, Config.EPSILON_MAX))

    def update(self):
        if len(self.buffer) < Config.BATCH_SIZE:
            return None, None

        s, a, r, s_next = self.buffer.sample()
        # Critic 更新
        q_curr = self.critic(s, a)
        with torch.no_grad():
            a_next   = self.actor_target(s_next)
            q_target = r + Config.GAMMA * self.critic_target(s_next, a_next)
        loss_q = nn.MSELoss()(q_curr, q_target)
        self.critic_optim.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # Actor 更新
        loss_pi = -self.critic(s, self.actor(s)).mean()
        self.actor_optim.zero_grad()
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()

        # 软更新
        with torch.no_grad():
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(Config.TAU * p.data + (1 - Config.TAU) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(Config.TAU * p.data + (1 - Config.TAU) * tp.data)

        return loss_q.item(), loss_pi.item()


# ----------------------
# 非线性平滑状态转移
# ----------------------
def transition_rrisk(rrisk, epsilon, noise_scale=0.03):
    target = (Config.EPSILON_MAX - epsilon) / (Config.EPSILON_MAX - Config.EPSILON_MIN)
    delta  = target - rrisk
    step   = Config.TRANS_ETA * np.sign(delta) * (abs(delta) ** Config.TRANS_GAMMA)
    new_rr = rrisk + step + np.random.normal(scale=noise_scale)
    return float(np.clip(new_rr, 0.0, 1.0))


# ----------------------
# 改进的奖励函数
# ----------------------
def calculate_reward(rrisk, epsilon):
    # 隐私增益
    logistic      = 1.0 / (1.0 + np.exp(-Config.PRIV_KAPPA * (rrisk - Config.PRIV_S0)))
    budget_factor = ((Config.EPSILON_MAX - epsilon) /
                     (Config.EPSILON_MAX - Config.EPSILON_MIN)) ** Config.PRIV_DELTA
    g_priv = Config.ALPHA * logistic * budget_factor

    # 效用损失
    util_penalty = Config.BETA * (1 - Config.UTIL_RHO * rrisk) * (Config.UTIL_SIGMA0 / epsilon) ** 2

    return float(g_priv - util_penalty)


# ----------------------
# 训练 & 可视化
# ----------------------
def train(episodes=600, steps_per_ep=25):
    agent = DDPGAgent()
    history = {
        'rrisk': [], 'epsilon': [], 'reward': [],
        'loss_q': [], 'loss_pi': []
    }

    for ep in range(episodes):
        rrisk = float(np.random.uniform(0.1, 0.9))
        agent.noise.reset()
        ep_rewards = []

        for _ in range(steps_per_ep):
            state   = np.array([rrisk], dtype=np.float32)
            epsilon = agent.select_action(state)
            reward  = calculate_reward(rrisk, epsilon)
            rrisk   = transition_rrisk(rrisk, epsilon)
            next_s  = np.array([rrisk], dtype=np.float32)

            agent.buffer.add(state, epsilon, reward, next_s)
            loss_q, loss_pi = agent.update()

            if loss_q is not None:
                history['loss_q'].append(loss_q)
                history['loss_pi'].append(loss_pi)

            history['rrisk'].append(rrisk)
            history['epsilon'].append(epsilon)
            history['reward'].append(reward)
            ep_rewards.append(reward)

        avg_reward = float(np.mean(ep_rewards))
        print(f"Episode {ep+1:03d} | Avg Reward: {avg_reward:7.2f} | Final Rrisk: {rrisk:.2f} | Final Epsilon: {epsilon:.2f}")

    return agent, history

def visualize(agent, history, max_steps=10000):
    # ———— 全局字体放大 ————
    plt.rcParams['font.size']        = 16
    plt.rcParams['axes.labelsize']   = 18
    plt.rcParams['axes.titlesize']   = 20
    plt.rcParams['xtick.labelsize']  = 15
    plt.rcParams['ytick.labelsize']  = 15
    plt.rcParams['legend.fontsize']  = 15
    plt.rcParams['lines.linewidth']  = 2.5
    plt.rcParams['grid.linewidth']   = 0.7
    plt.rcParams['grid.color']       = '#444444'

    # 深色调配色
    policy_color = '#003f5c'
    hist_color   = '#d62728'

    # —— 策略曲线（左图） ——
    plt.figure(figsize=(12, 5))  # 2 图结构
    plt.subplot(1, 2, 1)
    rr_all = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)
    with torch.no_grad():
        eps_pred = agent.actor(torch.FloatTensor(rr_all)).cpu().numpy().flatten()
    plt.plot(rr_all, eps_pred, color=policy_color)
    plt.xlabel('Risk')
    plt.ylabel('Epsilon')
    plt.title('Policy')
    plt.grid(True)

    # —— 奖励分布（右图） ——
    rewards = history['reward'][:max_steps]
    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=40, density=True, color=hist_color, alpha=0.8)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.xlim([-5, max(rewards) + 0.5])  # 设置横坐标从 -5 开始
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig('1_modified_policy_reward.png', dpi=150)

    # —— Actor & Critic 损失曲线（前 max_steps 步更新） ——
    loss_q  = history['loss_q'][:max_steps]
    loss_pi = history['loss_pi'][:max_steps]
    plt.figure(figsize=(8, 5))
    plt.plot(loss_q,  label='Critic Loss', color='#2ca02c')
    plt.plot(loss_pi, label='Actor Loss',  color='#d62728')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('2_dark_large.png', dpi=150)


def print_discrete_policy(agent):
    rr_values = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
    print("Risk\t->\tEpsilon")
    for r in rr_values:
        eps = agent.actor(torch.FloatTensor([[r]])).item()
        print(f"{r:.1f}\t->\t{eps:.3f}")


if __name__ == "__main__":
    start_time = time.time()
    agent, history = train(episodes=1000, steps_per_ep=25)
    elapsed_time = time.time() - start_time
    print(f"DDPG training completed in {elapsed_time:.2f} seconds.")

    # 保存 Actor 模型
    model_path = "actor_model.pth"
    torch.save(agent.actor.state_dict(), model_path)
    print(f"Actor model saved to {model_path}")

    visualize(agent, history,max_steps=10000)
    print_discrete_policy(agent)