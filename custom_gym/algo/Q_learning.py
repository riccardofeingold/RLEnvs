import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class QLearningCfg:
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.1


class QLearningAgent:
    def __init__(self, env: gym.Env, cfg: QLearningCfg):
        self.env = env

        self.alpha = cfg.learning_rate
        self.gamma = cfg.discount_factor
        self.epsilon = cfg.initial_epsilon
        self.min_epsilon = cfg.min_epsilon
        self.epsilon_decay = cfg.epsilon_decay

        self.q_table = defaultdict(lambda: np.zeros((env.action_space.n)))
        self.training_error = []

    def get_action(self, obs: dict) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[str(obs)]))

    def update(
        self, obs: dict, action: int, reward: float, terminated: bool, next_obs: dict
    ):
        next_obs_str = str(next_obs)
        obs_str = str(obs)
        future_q_table = (
            (not terminated) * self.gamma * np.max(self.q_table[next_obs_str])
        )
        temporal_diff = reward + future_q_table - self.q_table[obs_str][action]
        self.q_table[obs_str][action] += self.alpha * temporal_diff

        self.training_error.append(temporal_diff)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
