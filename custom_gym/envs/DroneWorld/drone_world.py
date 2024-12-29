import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv 

class DroneWorldEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "rgbd_tuple"
        ]
    }
    def __init__(
        self,
        xml_file: str = "skydio_x2/scene.xml",
        
    ):
        super().__init__()

        self.action_space = spaces.Box(0, 10, shape=(4,), dtype=float)
        self.observation_space = spaces.Box(0, 255, shape=(4, 100, 100), dtype=np.uint8)

    def _get_obs(self):
        return None
    
    def _get_info(self):
        return None

    def _get_reward(self):
        return None

    def step(self, action):
        observation = self._get_obs()
        reward = self._get_reward()
        terminated = False
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        observations = None
        info = None

        return observations, info
    
    def render(self):
        pass

    def close(self):
        pass