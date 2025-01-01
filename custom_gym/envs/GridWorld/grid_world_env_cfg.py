import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from custom_gym.envs.env_cfg import PygameSimulationCfg


@dataclass
class GridWorldRewardScales:
    has_reached_target: float = 1.0
    distance_to_target: float = 1.0


@dataclass
class GridWorldEnvCfg:
    field_size: int = np.array([5, 5])
    episode_length: int = 100_000
    observation_space = gym.spaces.Dict(
        {
            "agent": gym.spaces.Box(
                np.array([0, 0]),
                np.array([field_size[0] - 1, field_size[1] - 1]),
                shape=(2,),
                dtype=int,
            ),
            "target": gym.spaces.Box(
                np.array([0, 0]),
                np.array([field_size[0] - 1, field_size[1] - 1]),
                shape=(2,),
                dtype=int,
            ),
        }
    )
    action_space = gym.spaces.Discrete(4)

    reward_scales = GridWorldRewardScales()
    sim = PygameSimulationCfg()
