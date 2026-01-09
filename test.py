"""
测试训练好的TD3模型在2D无人机避障任务中的性能
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
from uav_navigation import Drone2DEnv, TD3Agent

def test_model(model_path, num_tests=100, render=False):
    """
    测试训练好的模型性能
    
    参数:
    model_path: 模型文件路径
    num_tests: 测试回合数
    render: 是否可视化
    
    返回:
    测试结果字典
    """
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建环境和agent
    env = Drone2DEnv(grid_size=15, num_obstacles=8, render_mode="human" if render else None)
    agent = TD3Agent(state_dim=env.observation_space.shape[0], action_dim=2, device=device)
    
    # 加载模型
    agent.load(model_path)
    
    # 测试结果统计
    results = {
        'success_count': 0,
        'collision_count': 0,
        'timeout_count': 0,
        'total_rewards': [],
        'total_steps': [],
        'final_distances': []
    }
    
    print(f"\n开始测试模型: {model_path}")
    print(f"测试回合数: {num_tests}")
    print("=" * 50)
    
    for i in range(1, num_tests + 1):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.select_action(state, noise_scale=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            ep_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        # 统计结果
        final_dist = np.linalg.norm(env.pos - env.target)
        results['final_distances'].append(final_dist)
        results['total_rewards'].append(ep_reward)
        results['total_steps'].append(steps)
        
        if final_dist < 1.0:
            results['success_count'] += 1
        elif truncated:
            results['timeout_count'] += 1
        elif terminated:
            results['collision_count'] += 1
        
        if i % 10 == 0:
            print(f"回合 {i}: 奖励 = {ep_reward:.2f}, 步数 = {steps}, 最终距离 = {final_dist:.2f}, 结果 = {'成功' if final_dist < 1.0 else '失败'}")
    
    # 计算统计指标
    avg_reward = np.mean(results['total_rewards'])
    avg_steps = np.mean(results['total_steps'])
    avg_distance = np.mean(results['final_distances'])
    success_rate = results['success_count'] / num_tests * 100
    
    # 打印测试报告
    print("\n" + "=" * 50)
    print("测试报告")
    print("=" * 50)
    print(f"成功率: {success_rate:.2f}% ({results['success_count']}/{num_tests})")
    print(f"碰撞率: {results['collision_count']/num_tests*100:.2f}% ({results['collision_count']}/{num_tests})")
    print(f"超时率: {results['timeout_count']/num_tests*100:.2f}% ({results['timeout_count']}/{num_tests})")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_steps:.2f}")
    print(f"平均最终距离: {avg_distance:.2f}")
    print(f"最大奖励: {np.max(results['total_rewards']):.2f}")
    print(f"最小奖励: {np.min(results['total_rewards']):.2f}")
    print("=" * 50)
    
    plt.close('all')
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试TD3无人机避障模型")
    parser.add_argument("--model", type=str, 
                      default="td3_drone_gpu_20260107_175015/best_model.pth",
                      help="模型文件路径")
    parser.add_argument("--num_tests", type=int, default=100,
                      help="测试回合数")
    parser.add_argument("--render", action="store_true",
                      help="是否可视化")
    
    args = parser.parse_args()
    
    # 测试指定模型
    test_model(args.model, args.num_tests, args.render)
