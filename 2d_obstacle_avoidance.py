"""
2D Drone Obstacle Avoidance with TD3 (GPU加速 + 剩余时间提示)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time   # 用于计算剩余时间

# ============================ 环境 ============================
class Drone2DEnv(gym.Env):
    def __init__(self, grid_size=15, num_obstacles=8, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_dim = 6 + grid_size * grid_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        for _ in range(self.num_obstacles):
            while True:
                x, y = np.random.randint(1, self.grid_size - 1, size=2)
                if not (x in [1, self.grid_size-2] and y in [1, self.grid_size-2]):
                    self.grid[x, y] = 1
                    break

        self.pos = np.array([1.0, 1.0], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.target = np.array([self.grid_size - 2.0, self.grid_size - 2.0], dtype=np.float32)

        self.steps = 0
        self.max_steps = 300
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.pos, self.vel, self.target, self.grid.flatten()]).astype(np.float32)

    def step(self, action):
        self.vel = np.clip(action, -1.0, 1.0)
        self.pos += self.vel
        self.pos = np.clip(self.pos, 0.0, self.grid_size - 1.0)
        self.steps += 1

        dist = np.linalg.norm(self.pos - self.target)
        grid_x, grid_y = int(self.pos[0]), int(self.pos[1])
        collided = self.grid[grid_x, grid_y] == 1

        reward = 0.0
        
        # 1. 距离奖励：靠近目标获得正奖励，远离获得负奖励
        prev_dist = getattr(self, 'prev_dist', dist)
        dist_change = prev_dist - dist  # 距离减少为正
        reward += dist_change * 1.5
        self.prev_dist = dist
        
        # 2. 添加目标方向向量奖励：鼓励朝目标中心飞行，而不是在目标附近徘徊
        target_direction = self.target - self.pos
        dist_to_target = np.linalg.norm(target_direction)
        if dist_to_target > 0.01:
            target_direction_normalized = target_direction / dist_to_target
            # 速度在目标方向上的分量
            direction_alignment = np.dot(self.vel, target_direction_normalized)
            reward += direction_alignment * 2.0  # 朝目标飞获得正奖励
        
        # 3. 目标中心接近奖励：距离目标越近，奖励越高（仅在距离<3.0时）
        if dist < 3.0:
            center_bonus = (3.0 - dist) * 2.0
            reward += center_bonus
        
        # 4. 避障奖励系统
        min_obstacle_dist = float('inf')
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 1:
                    obstacle_pos = np.array([float(i), float(j)])
                    d = np.linalg.norm(self.pos - obstacle_pos)
                    if d < min_obstacle_dist:
                        min_obstacle_dist = d
        
        if collided:
            reward -= 50.0  # 碰撞惩罚
        else:
            if min_obstacle_dist < 1.5:
                obstacle_penalty = (1.5 - min_obstacle_dist) * 8.0
                reward -= obstacle_penalty
                reward -= 0.05
            else:
                safety_bonus = min(0.5, (min_obstacle_dist - 1.5) * 0.3)
                reward += safety_bonus

        # 5. 目标到达奖励（必须真正到达）
        if dist < 1.0:
            reward += 80.0
            terminated = True
        elif dist < 2.0:
            reward += 5.0

        # 6. 增大步骤惩罚：强烈鼓励快速完成任务
        reward -= 0.1  # 从-0.03增大到-0.1

        # 7. 边界惩罚（增强）
        if self.pos[0] < 0.5 or self.pos[0] > self.grid_size - 1.5 or \
           self.pos[1] < 0.5 or self.pos[1] > self.grid_size - 1.5:
            reward -= 0.5  # 增强边界惩罚

        terminated = collided or (dist < 1.0)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return
        if not hasattr(self, "fig"):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.ax.clear()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 1:
                    self.ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))

        self.ax.add_patch(plt.Circle((1, 1), 0.4, color='blue', label='Start'))
        self.ax.add_patch(plt.Circle((self.target[1], self.target[0]), 0.4, color='green', label='Target'))
        self.ax.add_patch(plt.Circle((self.pos[1], self.pos[0]), 0.4, color='red', label='Drone'))
        self.ax.arrow(self.pos[1], self.pos[0], self.vel[1]*0.5, self.vel[0]*0.5,
                      head_width=0.2, head_length=0.3, fc='orange', ec='orange')

        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f"Step: {self.steps} | Dist: {np.linalg.norm(self.pos - self.target):.2f}")
        self.ax.legend(loc='upper right')
        plt.draw()
        plt.pause(0.05)

# ============================ TD3 Agent (GPU 支持) ============================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, action_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, 512), nn.ReLU(),
                                nn.Linear(512, 512), nn.ReLU(),
                                nn.Linear(512, 512), nn.ReLU(),
                                nn.Linear(512, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, 512), nn.ReLU(),
                                nn.Linear(512, 512), nn.ReLU(),
                                nn.Linear(512, 512), nn.ReLU(),
                                nn.Linear(512, 1))
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x), self.q2(x)
    def Q1(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x)

class TD3Agent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        print(f"使用设备: {self.device}")

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)  # 降低学习率

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)  # Critic学习率可以稍高

        self.buffer = deque(maxlen=500000)  # 增大经验回放池
        self.batch_size = 256  # 增大批次大小
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.updates_per_step = 2  # 每步环境交互训练2次
        self.total_steps = 0

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().detach().numpy()[0]
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, size=2)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def store(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, 1.0 - done))

    def train(self):
        if len(self.buffer) < self.batch_size: return
        self.total_steps += 1
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s_, not_done = map(np.array, zip(*batch))

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        not_done = torch.FloatTensor(not_done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(s_) + noise).clamp(-1.0, 1.0)
            q1, q2 = self.critic_target(s_, next_action)
            target_q = torch.min(q1, q2)
            target_q = r + not_done * self.gamma * target_q

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.total_steps % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

# ============================ 训练主函数（含剩余时间提示） ============================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}，将使用GPU加速训练！")
    else:
        print("未检测到GPU，将使用CPU训练（会比较慢）")

    env = Drone2DEnv(grid_size=15, num_obstacles=8)
    agent = TD3Agent(state_dim=env.observation_space.shape[0], action_dim=2, device=device)

    episodes = 1000
    rewards_history = []
    best_reward = -np.inf
    start_time = time.time()
    episode_times = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"td3_drone_gpu_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    print("开始训练...\n")

    for ep in range(1, episodes + 1):
        ep_start = time.time()
        state, _ = env.reset()
        ep_reward = 0
        done = False

        # 噪声随训练逐渐衰减
        noise_scale = max(0.05, 0.3 * (0.99 ** ep))

        while not done:
            action = agent.select_action(state, noise_scale)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, done)
            
            # 每步环境交互进行多次训练，充分利用GPU计算资源
            for _ in range(agent.updates_per_step):
                agent.train()
            
            state = next_state
            ep_reward += reward

        rewards_history.append(ep_reward)
        episode_times.append(time.time() - ep_start)

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(save_dir, "best_model.pth"))

        # === 每10个episode显示一次剩余时间 ===
        if ep % 10 == 0:
            avg_time_per_ep = np.mean(episode_times[-10:])
            remaining_episodes = episodes - ep
            eta_seconds = remaining_episodes * avg_time_per_ep
            eta_str = time.strftime("%H小时%M分钟%S秒", time.gmtime(eta_seconds))
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
            print(f"Episode {ep}/{episodes} | 近50轮平均奖励: {avg_reward:+.2f} | "
                  f"预计剩余时间: {eta_str} | 本轮用时: {episode_times[-1]:.1f}s")

    # 保存最终结果
    agent.save(os.path.join(save_dir, "final_model.pth"))
    plt.figure(figsize=(10,5))
    plt.plot(rewards_history, alpha=0.6)
    plt.plot(np.convolve(rewards_history, np.ones(50)/50, mode='valid'), 'r-', label='50-ep MA')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "reward_curve.png"))
    plt.close()

    total_time = time.strftime("%H小时%M分钟%S秒", time.gmtime(time.time() - start_time))
    print(f"\n训练完成！总耗时: {total_time}")
    print(f"所有文件已保存到文件夹: {save_dir}")
    return env, agent, save_dir

# ============================ 测试可视化 ============================
def test_visualize(agent, save_dir, num_tests=6):
    print(f"\n开始展示 {num_tests} 个测试回合...")
    agent.load(os.path.join(save_dir, "best_model.pth"))

    for i in range(1, num_tests + 1):
        env = Drone2DEnv(grid_size=15, num_obstacles=8, render_mode="human")
        state, _ = env.reset()
        done = False
        print(f"\n--- 测试回合 {i} ---")
        while not done:
            action = agent.select_action(state, noise_scale=0.0)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            env.render()
        dist = np.linalg.norm(env.pos - env.target)
        print(f"回合结束 | 最终距离: {dist:.2f} | {'成功' if dist < 1.0 else '失败'}")
        input("按 Enter 继续下一个回合...")
        plt.close('all')

if __name__ == "__main__":
    train_env, trained_agent, save_folder = train()
    test_visualize(trained_agent, save_folder, num_tests=6)