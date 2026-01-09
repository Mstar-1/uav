"""
UAV 2D 避障环境定义
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class Drone2DEnv(gym.Env):
    """
    2D无人机避障环境
    
    参数:
    - grid_size: 网格大小，默认为15x15
    - num_obstacles: 障碍物数量，默认为8
    - render_mode: 渲染模式，"human"为可视化模式
    """
    def __init__(self, grid_size=15, num_obstacles=8, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_dim = 6 + grid_size * grid_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态
        
        参数:
        - seed: 随机种子
        - options: 其他选项
        
        返回:
        - 初始观测值
        - 空字典（兼容gymnasium接口）
        """
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
        self.max_steps = 400
        self.pos_history = [self.pos.copy()]
        self.prev_dist = None
        return self._get_obs(), {}

    def _get_obs(self):
        """
        获取环境观测值
        
        返回:
        - 观测值向量，包含位置、速度、目标位置和网格地图
        """
        return np.concatenate([self.pos, self.vel, self.target, self.grid.flatten()]).astype(np.float32)

    def step(self, action):
        """
        执行动作并返回环境状态
        
        参数:
        - action: 无人机的动作向量（速度分量）
        
        返回:
        - next_state: 下一个状态观测值
        - reward: 奖励值
        - terminated: 是否终止（碰撞或到达目标）
        - truncated: 是否截断（超过最大步数）
        - info: 额外信息字典
        """
        self.vel = np.clip(action, -1.0, 1.0)
        self.pos += self.vel
        self.pos = np.clip(self.pos, 0.0, self.grid_size - 1.0)
        self.steps += 1

        dist = np.linalg.norm(self.pos - self.target)
        grid_x, grid_y = int(self.pos[0]), int(self.pos[1])
        collided = self.grid[grid_x, grid_y] == 1

        reward = 0.0
        
        # 距离奖励：距离越近奖励越高
        max_dist = np.linalg.norm([self.grid_size, self.grid_size])
        distance_reward = (max_dist - dist) / max_dist * 2.0
        reward += distance_reward
        
        # 目标到达奖励：到达目标给予高额奖励
        if dist < 1.0:
            reward += 200.0
        
        # 碰撞惩罚
        if collided:
            reward -= 150.0
        
        # 时间惩罚：鼓励尽快到达目标
        reward -= 0.05
        
        # 边界惩罚：避免靠近边界
        if self.pos[0] < 1.0 or self.pos[0] > self.grid_size - 2.0 or \
           self.pos[1] < 1.0 or self.pos[1] > self.grid_size - 2.0:
            reward -= 0.3
        
        # 停滞惩罚：防止原地不动
        self.pos_history.append(self.pos.copy())
        if len(self.pos_history) > 10:
            recent_positions = np.array(self.pos_history[-10:])
            avg_movement = np.mean(np.linalg.norm(np.diff(recent_positions, axis=0), axis=1))
            if avg_movement < 0.1:
                reward -= 3.0
        if len(self.pos_history) > 20:
            self.pos_history.pop(0)

        terminated = collided or (dist < 1.0)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        """
        渲染环境
        """
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
