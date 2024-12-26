import gymnasium as gym
import numpy as np

from envs.GridWorld.grid_world_env import GridWorldEnv

env = gym.make("envs:envs/GridWorld-v0", render_mode="human")

observations, info = env.reset()
for _ in range(100):
    env.render()
    action = env.action_space.sample()
    observations, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        env.reset()

env.close()