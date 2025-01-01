import gymnasium as gym
from custom_gym.envs.GridWorld import GridWorldEnv, GridWorldEnvCfg
from custom_gym.algo.Q_learning import QLearningAgent, QLearningCfg
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import json

SIMULATION_LENGTH = 300
MAX_STEPS = 10


@dataclass
class GridWorldQLearningCfg(QLearningCfg):
    learning_rate = 0.0
    epsilon_decay = 0.0
    min_epsilon = 0.0
    initial_epsilon = 0.0
    discount_factor = 0.0


env = gym.make("GridWorld-v0", cfg=GridWorldEnvCfg, testing=True)
agent = QLearningAgent(env=env, cfg=GridWorldQLearningCfg)

with open("policies/GridWorld/Q_learning/q_values.json", "r") as file:
    dict_q_values = json.load(file)

agent.q_table = defaultdict(
    lambda: np.zeros(env.action_space.n),
    {key: np.array(value) for key, value in dict_q_values.items()},
)

for epsiode in range(SIMULATION_LENGTH):
    obs, info = env.reset()
    done = False
    counter = 0
    while not done:
        counter += 1
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        if counter > MAX_STEPS:
            done = True
        obs = next_obs
        env.render()


env.close()
