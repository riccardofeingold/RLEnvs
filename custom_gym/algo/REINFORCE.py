import torch
from torch import nn
import numpy as np
from dataclasses import dataclass

from custom_gym.networks.MLP import MLP

@dataclass
class REINFORCECfg:
    learning_rate: float = 1.0e-3
    discount_factor: float = 0.95
    epsilon: float = 1.0e-6 # small number to guarantee numerical stability

class REINFORCE:
    def __init__(self, obs_space_dims: int, action_space_dims: int, device: torch.device = "cpu", cfg = REINFORCECfg()):
        self.device = device

        self.learning_rate = cfg.learning_rate
        self.gamma = cfg.discount_factor
        self.eps = cfg.epsilon

        self.probs = []
        self.rewards = []

        self.net = MLP(
            input_size=obs_space_dims,
            hidden_layer_sizes=[256, 256, 256],
            activation_func=[nn.ReLU(), nn.ReLU(), nn.ReLU()],
            output_size=action_space_dims,
            with_stdv=True
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
    
    def sample_action(self, obs: np.ndarray) -> np.ndarray:
        obs = torch.tensor(np.array([obs]), device=self.device)
        action_means, action_stddevs = self.net(obs)
        if torch.sum(torch.isinf(action_means)) > 0.0 or torch.sum(torch.isinf(action_stddevs)) > 0.0:
            print("Means: ", action_means)
            print("STDV: ", action_stddevs)
            print("Obs: ", obs)
            torch.save(self.net, "./log/inf_fail.pth")
        if torch.sum(torch.isnan(action_means)) > 0.0 or torch.sum(torch.isnan(action_stddevs)) > 0.0:
            print("Means: ", action_means)
            print("STDV: ", action_stddevs)
            print("Obs: ", obs)
            torch.save(self.net, "./log/nan_fail.pth")

        eps = torch.ones_like(action_means, device=self.device) * self.eps
        action_stddevs = torch.clamp(action_stddevs, min=0, max=10)
        distribution = torch.distributions.Normal(action_means[0] + eps, action_stddevs[0] + eps)
        action = distribution.sample()
        prob = distribution.log_prob(action)
        self.probs.append(prob)

        action = torch.clamp(action, min=0, max=6.0)
        action = np.asarray(action.detach().cpu().numpy()).flatten()

        return action
    
    def update(self):
        running_g = 0.0
        gs = []

        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs, device=self.device) 
        log_probs = torch.stack(self.probs).to(self.device)
        log_probs_mean = log_probs.mean()

        loss = -torch.sum(log_probs_mean * deltas)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []
