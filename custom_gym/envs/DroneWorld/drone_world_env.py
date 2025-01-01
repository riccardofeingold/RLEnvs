import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from custom_gym.envs.DroneWorld.drone_world_env_cfg import DroneWorldEnvCfg
from custom_gym.utils.math_helpers import gaussian_distribution


class DroneWorldEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(self, cfg: DroneWorldEnvCfg):
        # only needed if I want to serialize or deserialize the class using the library pickle
        utils.EzPickle.__init__(self, cfg)

        self.cfg = cfg

        MujocoEnv.__init__(
            self,
            model_path=cfg.robot.xml_file,
            frame_skip=cfg.sim.frame_skip,
            observation_space=None,
            default_camera_config=cfg.sim.default_camera_config,
            render_mode=cfg.sim.render_mode,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.elapsed_time = 0.0
        self.action_space = cfg.action_space
        self.observation_space = cfg.observation_space
        self.reference_position = self._get_commands()

    @property
    def has_terminated(self) -> bool:
        contact_forces = self.data.cfrc_ext[1, 3:]
        if (
            np.linalg.norm(contact_forces)
            > self.cfg.robot.max_contact_force_with_ground
        ):
            return True
        return False

    def _get_obs(self, action):
        actuator_vel = self.data.actuator_velocity.flatten()
        sensordata = self.data.sensordata.flatten()
        lin_acc_b = sensordata[:3]
        ang_vel_b = sensordata[3:6]

        return np.concatenate((actuator_vel, lin_acc_b, ang_vel_b, action))

    def _get_info(self) -> dict:
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()

        body_position = qpos[:3]
        body_attitude_quat = qpos[3:]
        body_linear_vel = qvel[:3]
        body_angular_vel = qvel[3:]

        info = {
            "position_b": body_position,
            "attitude_b": body_attitude_quat,
            "linear_vel_b": body_linear_vel,
            "angular_vel_b": body_angular_vel,
        }

        return info

    def _get_commands(self) -> np.ndarray[np.float64]:
        return self.np_random.uniform(
            low=self.cfg.commands.reference_position_range[:, 0],
            high=self.cfg.commands.reference_position_range[:, 1],
        )

    def _get_reward(self, obs: np.ndarray[np.float64]) -> float:
        position_tracking_reward = (
            gaussian_distribution(
                diff=np.linalg.norm(self.data.qpos[:3] - self.reference_position),
                mu=self.cfg.reward_scales.position_reference_tracking["mean"],
                sigma=self.cfg.reward_scales.position_reference_tracking["variance"],
            )
            * self.cfg.reward_scales.position_reference_tracking["scale"]
        )

        slow_roll_pitch_rate_reward = (
            gaussian_distribution(
                diff=np.linalg.norm(obs[6:]),
                mu=self.cfg.reward_scales.slow_roll_pitch_rate["mean"],
                sigma=self.cfg.reward_scales.slow_roll_pitch_rate["variance"],
            )
            * self.cfg.reward_scales.slow_roll_pitch_rate["scale"]
        )

        rewards = slow_roll_pitch_rate_reward + position_tracking_reward
        reward_info = {
            "position_tracking_reward": position_tracking_reward,
            "slow_roll_pitch_rate_reward": slow_roll_pitch_rate_reward,
        }
        return rewards, reward_info

    def step(self, action):
        self.do_simulation(action, self.cfg.sim.frame_skip)

        observation = self._get_obs(action)
        reward, reward_info = self._get_reward(observation)
        terminated = self.has_terminated
        info = self._get_info()
        info.update(reward_info)

        if self.render_mode == "human":
            self.render()

        self.elapsed_time += self.dt
        truncated = (
            True if self.elapsed_time >= self.cfg.max_episode_length_s else False
        )

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos
        qpos[:3] = self.init_qpos[:3] + self.np_random.uniform(
            low=self.cfg.robot.initial_position[:, 0],
            high=self.cfg.robot.initial_position[:, 1],
        )

        qvel = self.init_qvel
        qvel[:3] = self.init_qvel[:3] + self.np_random.uniform(
            low=self.cfg.robot.initial_velocity[:, 0],
            high=self.cfg.robot.initial_velocity[:, 1],
        )
        self.set_state(qpos, qvel)

        self.reference_position = self._get_commands()
        observations = self._get_obs(np.zeros((4,)))

        self.elapsed_time = 0.0
        return observations

    def _get_reset_info(self):
        return self._get_info()
