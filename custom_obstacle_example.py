#!/usr/bin/env python3
"""
自定义障碍物示例
该脚本展示了如何在gym-pybullet-drones环境中创建自定义障碍物用于路径规划
"""

import os
import sys
import numpy as np
import pybullet as p
import gymnasium as gym
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import BaseSingleAgentAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics

class CustomObstacleAviary(BaseSingleAgentAviary):
    """自定义障碍物环境，用于路径规划"""
    
    def __init__(self, **kwargs):
        """初始化环境"""
        super().__init__(**kwargs)
        
    def _addObstacles(self):
        """添加自定义障碍物
        
        可以添加各种形状、大小和位置的障碍物，用于路径规划测试
        """
        # 调用父类方法（可选）
        # super()._addObstacles()
        
        # 使用pybullet内置的基本形状函数
        
        # 立方体障碍物
        def create_box(position, size):
            col_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=size, physicsClientId=self.CLIENT)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_box, 
                             basePosition=position, baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                             physicsClientId=self.CLIENT)
        
        # 圆柱体障碍物
        def create_cylinder(position, radius, height):
            col_cylinder = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=self.CLIENT)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_cylinder, 
                             basePosition=position, baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                             physicsClientId=self.CLIENT)
        
        # 球体障碍物
        def create_sphere(position, radius):
            col_sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.CLIENT)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_sphere, 
                             basePosition=position, baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                             physicsClientId=self.CLIENT)
        
        # 示例1：添加立方体障碍物 (1x1x1)
        create_box([1, 0, 0.5], [0.5, 0.5, 0.5])
        
        # 示例2：添加球体障碍物 (半径0.5)
        create_sphere([-1, 0, 0.5], 0.5)
        
        # 示例3：添加圆柱体障碍物 (半径0.5，高度1.0)
        create_cylinder([0, 1, 0.5], 0.5, 1.0)
        
        # 示例4：添加自定义尺寸的立方体 (0.5x0.5x0.5)
        create_box([0, -1, 0.5], [0.25, 0.25, 0.25])
        
        # 示例5：添加多个障碍物形成路径
        # 创建一个L形障碍物路径 (0.5x0.5x0.2)
        for i in range(5):
            create_box([2, -2 + i*0.5, 0.1], [0.25, 0.25, 0.1])
            
        for i in range(5):
            create_box([2 - i*0.5, 0, 0.1], [0.25, 0.25, 0.1])
    
    def _computeReward(self):
        """自定义奖励函数，鼓励无人机避开障碍物并到达目标位置"""
        state = self._getDroneStateVector(0)
        position = state[0:3]
        
        # 目标位置 [2, 2, 2]
        target_pos = np.array([2, 2, 2])
        
        # 计算到目标位置的距离
        distance_to_target = np.linalg.norm(position - target_pos)
        
        # 基础奖励：距离目标越近，奖励越高
        reward = -distance_to_target
        
        # 障碍物碰撞惩罚
        # 这里可以添加碰撞检测逻辑，与障碍物碰撞时给予惩罚
        # 简化版本：如果无人机飞得太低（可能碰撞地面或障碍物），给予惩罚
        if position[2] < 0.2:
            reward -= 10
            
        return reward
    
    def _computeDone(self):
        """自定义终止条件"""
        state = self._getDroneStateVector(0)
        position = state[0:3]
        
        # 目标位置 [2, 2, 2]
        target_pos = np.array([2, 2, 2])
        
        # 如果到达目标位置附近（距离小于0.5），任务完成
        if np.linalg.norm(position - target_pos) < 0.5:
            return True
        
        # 如果无人机飞得太低（可能碰撞地面），任务失败
        if position[2] < 0.1:
            return True
            
        # 如果超过最大时间步，任务结束
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
            
        return False
    
    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.
        
        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.
        
        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.
        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                              clipped_pos_xy,
                                              clipped_pos_z,
                                              clipped_rp,
                                              clipped_vel_xy,
                                              clipped_vel_z
                                              )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    def _clipAndNormalizeStateWarning(self, state, clipped_pos_xy, clipped_pos_z, clipped_rp, clipped_vel_xy, clipped_vel_z):
        """Debugging printouts associated to `_clipAndNormalizeState`.
        
        Print a warning if values in a state vector is out of the clipping range.
        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in CustomObstacleAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in CustomObstacleAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in CustomObstacleAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in CustomObstacleAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in CustomObstacleAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
    
    def _computeInfo(self):
        """Computes the current info dict(s).
        
        Unused.
        
        Returns
        -------
        dict[str, int]
            Dummy value.
        """
        return {"answer": 42}

def main():
    """主函数"""
    # 创建自定义环境
    env = CustomObstacleAviary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=np.array([[0, 0, 0.5]]),  # 初始位置（numpy数组）
        initial_rpys=np.array([[0, 0, 0]]),    # 初始姿态（numpy数组）
        physics=Physics.PYB,         # 物理引擎
        freq=240,                    # 模拟频率
        aggregate_phy_steps=1,       # 物理更新步数
        gui=True,                    # 显示GUI
        record=False,                # 不录制视频
        obs=ObservationType.KIN,     # 观察类型：运动学信息
        act=ActionType.RPM           # 动作类型：直接控制电机转速
    )
    
    # 重置环境
    obs = env.reset()
    
    # 运行模拟
    for i in range(10*env.SIM_FREQ):  # 运行10秒
        # 简单的测试动作：悬停
        action = np.array([5000, 5000, 5000, 5000])  # 电机转速
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        # 渲染环境
        env.render()
        
        # 检查是否结束
        if done:
            print(f"Episode finished after {i} steps")
            obs = env.reset()
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
