"""
TD3 智能体模型定义
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import random
from collections import deque

class Actor(nn.Module):
    """
    Actor 网络：将状态映射到连续动作空间
    
    参数:
    - state_dim: 状态维度
    - action_dim: 动作维度
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),  # 添加dropout正则化
            nn.Linear(512, action_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    """
    Critic 网络：估计动作价值
    包含两个Q网络以减少过估计偏差
    
    参数:
    - state_dim: 状态维度
    - action_dim: 动作维度
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # 添加dropout正则化
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # 添加dropout正则化
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # 添加dropout正则化
            nn.Linear(512, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # 添加dropout正则化
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # 添加dropout正则化
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # 添加dropout正则化
            nn.Linear(512, 1)
        )
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x), self.q2(x)
    def Q1(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x)

class TD3Agent:
    """
    TD3 (Twin Delayed Deep Deterministic Policy Gradient) 智能体
    
    参数:
    - state_dim: 状态维度
    - action_dim: 动作维度
    - device: 计算设备 (CPU/GPU)
    """
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        print(f"使用设备: {self.device}")

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)  # 适当提高Actor学习率
        self.actor_scheduler = ExponentialLR(self.actor_opt, gamma=0.9995)  # 更慢的衰减率

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)  # 调整Critic学习率
        self.critic_scheduler = ExponentialLR(self.critic_opt, gamma=0.9990)  # 更慢的衰减率

        self.buffer = deque(maxlen=300000)  # 适当增大经验回放池
        self.batch_size = 256  # 增大批次大小
        self.gamma = 0.99
        self.tau = 0.001  # 保持软更新率
        self.policy_noise = 0.2  # 适当增加策略噪声
        self.noise_clip = 0.5  # 适当增加噪声裁剪
        self.policy_delay = 2  # 调整策略延迟为2步
        self.updates_per_step = 2  # 每步环境交互训练2次
        self.total_steps = 0

    def select_action(self, state, noise_scale=0.1):
        """
        根据当前状态选择动作
        
        参数:
        - state: 当前状态
        - noise_scale: 噪声强度，用于探索
        
        返回:
        - 动作向量
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # 设置为评估模式
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().detach().numpy()[0]
        # 恢复训练模式
        self.actor.train()
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def store(self, s, a, r, s_, done):
        """
        存储经验到回放缓冲池
        
        参数:
        - s: 当前状态
        - a: 动作
        - r: 奖励
        - s_: 下一个状态
        - done: 是否终止
        """
        self.buffer.append((s, a, r, s_, 1.0 - done))

    def train(self):
        """
        从回放缓冲池中采样并训练智能体
        """
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
            self.actor_scheduler.step()  # 学习率衰减

            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            
            # Critic也进行学习率衰减
            self.critic_scheduler.step()

    def save(self, path):
        """
        保存模型参数
        
        参数:
        - path: 模型保存路径
        """
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)

    def load(self, path):
        """
        加载模型参数
        
        参数:
        - path: 模型加载路径
        """
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        print(f"模型已加载: {path}")
