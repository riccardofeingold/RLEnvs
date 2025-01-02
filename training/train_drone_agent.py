import torch
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from custom_gym.envs import DroneWorld
from custom_gym.envs.DroneWorld.drone_world_env_cfg import DroneWorldEnvCfg
from custom_gym.algo.REINFORCE import REINFORCE

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = DroneWorldEnvCfg()
cfg.sim.render_mode = None
cfg.epsiode_length = 200_000

env = gym.make(id="DroneWorld-v0", cfg=cfg)
agent = REINFORCE(cfg.observation_space.shape[0], cfg.action_space.shape[0], device)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 100)

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
    agent.update()

    if episode % 500 == 0:
        avg_reward = np.mean(wrapped_env.return_queue)
        print("Episode: ", episode, " Average Reward", avg_reward)
        torch.save(agent.net, f"./policies/DroneWorld/REINFORCE/hoverPolicy_{episode}.pth")

torch.save(agent.net, "./policies/DroneWorld/REINFORCE/hoverPolicy_final.pth")

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