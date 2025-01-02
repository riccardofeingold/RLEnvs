import torch
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from custom_gym.envs import DroneWorld
from custom_gym.envs.DroneWorld.drone_world_env_cfg import DroneWorldEnvCfg
from custom_gym.algo.REINFORCE import REINFORCE

import numpy as np
import os
import re

device = "cpu"

cfg = DroneWorldEnvCfg()
cfg.sim.render_mode = "human"
cfg.sim.frame_skip = 1
cfg.epsiode_length = 200_000

env = gym.make(id="DroneWorld-v0", cfg=cfg)
agent = REINFORCE(cfg.observation_space.shape[0], cfg.action_space.shape[0], device)

folder_path = "./policies/DroneWorld/REINFORCE/"
pattern = r"hoverPolicy_(\d+)\.pth"
highest_episode_file = None
highest_episode = -1

for file in os.listdir(folder_path):
    match = re.match(pattern, file)
    if match:
        episode = int(match.group(1))  # Extract episode number
        if episode > highest_episode:
            highest_episode = episode
            highest_episode_file = file

agent.net = torch.load(f"./policies/DroneWorld/REINFORCE/{highest_episode_file}").to(device)
# agent.net = torch.load("./policies/DroneWorld/REINFORCE/hoverPolicy_121000.pth").to(device)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

reward_over_episodes = []
for episode in tqdm(range(cfg.epsiode_length)):
    observation, info = wrapped_env.reset()
    done = False

    while not done:
        action = agent.sample_action(observation)
        observation, rewards, terminated, truncated, info = wrapped_env.step(action)

        agent.rewards.append(rewards)

        done = terminated or truncated
 
    reward_over_episodes.append(wrapped_env.return_queue[-1])

    if episode % 500 == 0:
        avg_reward = int(np.mean(wrapped_env.return_queue))
        print("Episode: ", episode, " Average Reward", avg_reward)
        torch.save(agent.net, f"./policies/DroneWorld/REINFORCE/hoverPolicy_{episode}.pth")

# Plot the rewards
episodes = list(range(1, cfg.epsiode_length + 1))
plt.figure(figsize=(10, 6))
plt.plot(episodes, reward_over_episodes, label='Reward per Episode', color='blue')

# Optional: Calculate and plot a moving average for smoothing
window_size = 10
moving_avg = np.convolve(reward_over_episodes, np.ones(window_size)/window_size, mode='valid')
plt.plot(episodes[:len(moving_avg)], moving_avg, label='Moving Average (10 episodes)', color='red')

# Add labels, title, and legend
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards Collected Through Episodes')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()