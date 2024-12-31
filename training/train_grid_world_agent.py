from dataclasses import dataclass
import json
from tqdm import tqdm
import gymnasium as gym
from custom_gym.envs.GridWorld import GridWorldEnv, GridWorldEnvCfg
from custom_gym.algo.Q_learning import QLearningAgent, QLearningCfg
import numpy as np

EPISODES = 100_000

@dataclass
class GridWorldQLearningCfg(QLearningCfg):
    learning_rate = 0.01
    discount_factor = 0.95
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (EPISODES / 2)
    min_epsilon = 0.1

env = gym.make(
    "GridWorld-v0", 
    cfg=GridWorldEnvCfg
)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=EPISODES)

agent = QLearningAgent(
    env=env,
    cfg=GridWorldQLearningCfg 
)

for episode in tqdm(range(GridWorldEnvCfg.episode_length)):
    obs, info = env.reset()
    
    done = False
    while not done:
        action = agent.get_action(obs)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncated
        obs = next_obs
        # env.render()
    
    agent.decay_epsilon()
# env.close()

dict_q_values = {key: value.tolist() for key, value in agent.Q_values.items()}
with open('policies/GridWorld/Q_learning/q_values.json', 'w') as file:
    json.dump(dict_q_values, file)

from matplotlib import pyplot as plt
# visualize the episode rewards, episode length and training error in one figure
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

# np.convolve will compute the rolling mean for 100 episodes
WINDOW_SIZE = 10
axs[0].plot(np.convolve(env.return_queue, np.ones(WINDOW_SIZE)))
axs[0].set_title("Episode Rewards")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

axs[1].plot(np.convolve(env.length_queue, np.ones(WINDOW_SIZE)))
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

axs[2].plot(np.convolve(agent.training_error, np.ones(WINDOW_SIZE)))
axs[2].set_title("Training Error")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Temporal Difference")

plt.tight_layout()
plt.show()