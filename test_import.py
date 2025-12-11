# 简单测试脚本：检查包导入是否正常
import gymnasium as  gym
import numpy as np

try:
    # 测试导入gym_pybullet_drones的主要模块
    from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.utils import sync
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.DroneModel import DroneModel
    
    print("✅ 成功导入所有必要模块")
    
    # 测试创建环境（不渲染以避免图形界面问题）
    env = HoverAviary(drone_model=DroneModel.CF2X,
                      initial_xyzs=np.array([[0, 0, 0.5]]),
                      gui=False,  # 不开启渲染
                      record=False)
    
    print("✅ 成功创建环境")
    
    # 测试重置环境
    obs = env.reset()
    print("✅ 成功重置环境")
    print(f"观测空间维度: {obs['state'].shape}")
    
    # 测试关闭环境
    env.close()
    print("✅ 成功关闭环境")
    
    print("\n🎉 所有测试通过！gym_pybullet_drones包可以正常使用。")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()