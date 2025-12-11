# 极简验证脚本：仅测试基本导入
print("开始验证gym_pybullet_drones安装...")

try:
    # 仅测试核心包的导入
    from gym_pybullet_drones import __version__
    print(f"✅ 成功导入gym_pybullet_drones，版本: {__version__}")
    
    from gym_pybullet_drones.utils.DroneModel import DroneModel
    print("✅ 成功导入DroneModel")
    print(f"  可用无人机模型: {[m.name for m in DroneModel]}")
    
    print("\n🎉 验证完成！gym_pybullet_drones已成功安装并可导入。")
    
except Exception as e:
    print(f"❌ 验证失败: {e}")
    import traceback
    traceback.print_exc()