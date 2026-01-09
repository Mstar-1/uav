"""
使用TD3算法训练2D无人机避障智能体
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
from uav_navigation import Drone2DEnv, TD3Agent

def train():
    """
    训练TD3智能体
    
    返回:
    - 训练环境
    - 训练好的智能体
    - 保存目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}，将使用GPU加速训练！")
    else:
        print("未检测到GPU，将使用CPU训练（会比较慢）")

    env = Drone2DEnv(grid_size=15, num_obstacles=6)
    agent = TD3Agent(state_dim=env.observation_space.shape[0], action_dim=2, device=device)

    episodes = 2000
    rewards_history = []
    best_reward = -np.inf
    start_time = time.time()
    episode_times = []
    
    # 初始化成功率统计变量
    total_episodes_reached_target = 0
    episodes_reached_target_50 = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"td3_drone_gpu_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    print("开始训练...\n")

    for ep in range(1, episodes + 1):
        ep_start = time.time()
        state, _ = env.reset()
        ep_reward = 0
        done = False

        # 改进探索策略：使用简单而有效的高斯噪声，缓慢衰减
        # 基础噪声：初始值1.0，缓慢衰减到0.1
        base_noise = max(0.1, 1.0 * (0.995 ** ep))
        # 每10个回合增加一次探索强度
        if ep % 10 == 0:
            periodic_exploration = 0.5
        else:
            periodic_exploration = 0.0
        # 综合噪声
        noise_scale = base_noise + periodic_exploration

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

        # 检查是否成功到达终点
        final_dist = np.linalg.norm(env.pos - env.target)
        if final_dist < 1.0:
            total_episodes_reached_target += 1
            episodes_reached_target_50 += 1

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
        
        # === 每50个episode计算并打印成功率 ===
        if ep % 50 == 0:
            success_rate_50 = episodes_reached_target_50 / 50 * 100
            total_success_rate = total_episodes_reached_target / ep * 100
            print(f"\n=== 第 {ep-49} 到 {ep} 轮统计 ===")
            print(f"到达终点次数: {episodes_reached_target_50} / 50")
            print(f"成功率: {success_rate_50:.1f}%")
            print(f"累计总成功率: {total_success_rate:.1f}%")
            print(f"总到达次数: {total_episodes_reached_target} / {ep}")
            print("="*35)
            # 重置50轮统计
            episodes_reached_target_50 = 0

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

def test_visualize(agent, save_dir, num_tests=6):
    """
    可视化测试训练好的智能体
    
    参数:
    - agent: 训练好的智能体
    - save_dir: 模型保存目录
    - num_tests: 测试回合数
    """
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
