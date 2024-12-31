import gymnasium as gym
from tqdm import tqdm
from custom_gym.envs import DroneWorld
from custom_gym.envs.DroneWorld.drone_world_env_cfg import DroneWorldEnvCfg

import numpy as np

cfg = DroneWorldEnvCfg()
env = gym.make(
    id="DroneWorld-v0",
    cfg=cfg
)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

reward_over_episodes = []
for episode in tqdm(range(cfg.epsiode_length)):
    observation, info = wrapped_env.reset()
    done = False

    while not done:
        action = np.zeros((4,))
        observation, rewards, terminated, truncated, info = wrapped_env.step(action)

        done = terminated or truncated
    
    reward_over_episodes.append(wrapped_env.return_queue[-1])

    if episode % 1000 == 0:
        avg_reward = int(np.mean(wrapped_env.return_queue))
        print("Episode: ", episode, " Average Reward", avg_reward)

