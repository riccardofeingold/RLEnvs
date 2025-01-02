from gymnasium.spaces import Box
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

import numpy as np
from dataclasses import dataclass, field
from typing import List
from custom_gym.envs.env_cfg import MujocoSimulationCfg


@dataclass
class SensorCfg:
    gyro: str = "body_gyro"
    accelerometer: str = "body_linacc"
    global_body_attitude: str = "body_quat"
    camera: str = "track"


@dataclass
class DroneCfg:
    xml_file: str = "./custom_gym/assets/DroneWorld/skydio_x2/scene.xml"
    mass_range: np.ndarray[np.float64] = np.array([0.2, 1.0])
    initial_position: np.dtype = np.array([[0.0, 0.0], [0.0, 0.0], [0.3, 4.0]])
    initial_velocity: np.dtype = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    max_contact_force_with_ground: float = 1.0
    num_rotors: int = 4
    distance_between_two_opposite_rotors: float = 0.2
    max_thrust_range: np.ndarray[np.float64] = np.array([1.0, 6.0])
    max_joint_vel_range: np.ndarray[np.float64] = np.array([100, np.inf])
    body: str = "x2"
    actuators: List[str] = field(
        default_factory=lambda: [f"thrust{i}" for i in range(1, 5)]
    )
    sensors = SensorCfg()


@dataclass
class DroneHoverRewardScaleCfg:
    position_reference_tracking: dict = field(
        default_factory=lambda: {
            "scale": 2.0,
            "mean": 0.0,
            "variance": 1.0,
        }
    )
    slow_roll_pitch_rate: dict = field(
        default_factory=lambda: {"scale": 1.0, "mean": 0.0, "variance": 1.0}
    )
    low_lin_acc: dict = field(
        default_factory=lambda: {"scale": 1.0, "mean": 0.0, "variance": 1.0}
    )


@dataclass
class DroneHoverCommandsCfg:
    reference_position_range: np.ndarray[np.float64] = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 4.0]]
    )
    reference_velocity_range: np.ndarray[np.float64] = np.array(
        [[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]]
    )


@dataclass
class DroneWorldEnvCfg:
    robot = DroneCfg()
    sim = MujocoSimulationCfg()
    sim.render_mode = "human"
    sim.frame_skip = 5
    commands = DroneHoverCommandsCfg()

    epsiode_length: int = 100_000
    max_episode_length_s: float = 20.0
    action_scale: float = 1.0
    decimation: int = 4
    action_space = Box(
        low=0, high=robot.max_thrust_range[1], shape=(4,), dtype=np.float64
    )
    observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64)
    reward_scales = DroneHoverRewardScaleCfg()
