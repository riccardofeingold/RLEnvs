from collections import defaultdict
import gymnasium as gym
import numpy as np

class GridWorldAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float
    ):
        self.env = env
        self.Q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    
    def get_action(self, obs: dict) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.Q_values[str(obs)]))

    def update(
        self,
        obs: dict,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: dict
    ):
        next_obs_str = str(next_obs)
        obs_str = str(obs)
        future_q_values = (not terminated) * self.gamma * np.max(self.Q_values[next_obs_str])
        temporal_diff = reward + future_q_values - self.Q_values[obs_str][action] 
        self.Q_values[obs_str][action] = self.Q_values[obs_str][action] + self.lr * temporal_diff

        self.training_error.append(temporal_diff)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        pass