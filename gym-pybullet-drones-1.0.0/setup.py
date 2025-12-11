from setuptools import setup, find_packages

setup(name='gym_pybullet_drones',
    version='0.6.0',
    packages=find_packages(include=['gym_pybullet_drones', 'gym_pybullet_drones.*']),
    install_requires=['numpy', 'Pillow', 'matplotlib', 'cycler', 'gym', 'pybullet', 'stable_baselines3', 'ray[rllib]']
)
