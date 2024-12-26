import json
from tqdm import tqdm
import gymnasium as gym
import numpy as np

from agents.GridWorld.grid_world_agent import GridWorldAgent

LR = 0.01
Eps = 1.0
EPISODES = 100_000
ED = Eps / (EPISODES / 2)
FE = 0.1
DF = 0.95

env = gym.make(
    "envs:envs/GridWorld-v0", 
    # render_mode="human", 
    render_fps=120)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=EPISODES)

agent = GridWorldAgent(
    env=env,
    learning_rate=LR,
    initial_epsilon=Eps,
    epsilon_decay=ED,
    final_epsilon=FE,
    discount_factor=DF
)

for episode in tqdm(range(EPISODES)):
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