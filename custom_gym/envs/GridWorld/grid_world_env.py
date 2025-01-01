import gymnasium as gym
import numpy as np
import pygame
from enum import Enum
from custom_gym.envs.GridWorld.grid_world_env_cfg import GridWorldEnvCfg


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, cfg: GridWorldEnvCfg, testing: bool = False):
        if testing:
            cfg.sim.render_mode = "human"
            cfg.reward_scales.distance_to_target = 0.0
            cfg.reward_scales.has_reached_target = 0.0

        self.cfg = cfg
        self.size = cfg.field_size
        self.window_size = cfg.sim.window_size

        # Possible actions are: right, up, down, left
        self.action_space = cfg.action_space
        # Observations = position of player and position of target
        self.observation_space = cfg.observation_space
        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0], dtype=int),
            Actions.UP.value: np.array([0, 1], dtype=int),
            Actions.LEFT.value: np.array([-1, 0], dtype=int),
            Actions.DOWN.value: np.array([0, -1], dtype=int),
        }

        self.rewards = {
            "target_reached": lambda: (
                1 * cfg.reward_scales.has_reached_target
                if self._has_terminated()
                else 0
            ),
        }

        assert (
            cfg.sim.render_mode is None
            or cfg.sim.render_mode in self.metadata["render_modes"]
        )
        self.render_mode = cfg.sim.render_mode
        self.render_fps = (
            self.metadata["render_fps"] if cfg.sim.fps is None else cfg.sim.fps
        )

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        """
        Manhattan distance
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # seed self.np_random
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _has_terminated(self):
        return np.array_equal(self._agent_location, self._target_location)

    def step(self, action: Actions):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = self._has_terminated()
        reward = sum(f() for f in self.rewards.values())
        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size_width = self.window_size / self.size[0]
        pix_square_size_height = self.window_size / self.size[1]

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                np.array([pix_square_size_width, pix_square_size_height])
                * self._target_location,
                (pix_square_size_width, pix_square_size_height),
            ),
        )
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5)
            * np.array([pix_square_size_width, pix_square_size_height]),
            pix_square_size_width / 3,
        )

        for x in range(self.size[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size_width * x, 0),
                (pix_square_size_width * x, self.window_size),
                width=3,
            )

        for x in range(self.size[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size_height * x),
                (self.window_size, pix_square_size_height * x),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.render_fps)
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
