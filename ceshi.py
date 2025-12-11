import sys
print("开始验证gym_pybullet_drones安装...")

try:
    from gym_pybullet_drones.envs.BaseAviary import BaseAviary
    print("✅ 导入 BaseAviary 成功！安装验证通过。")
except ImportError as e:
    print(f"❌ 验证失败: {e}")
    sys.exit(1)