from typing import Dict, Union
from dataclasses import dataclass, field


@dataclass
class PygameSimulationCfg:
    fps: int = 4
    render_mode = None
    window_size: int = 512


@dataclass
class MujocoSimulationCfg:
    frame_skip: int = 5  # number of mujoco simulation steps per gym 'step()'
    render_mode: str = None
    default_camera_config: Dict[str, Union[float, int]] = field(
        default_factory=lambda: {"distance": 4.0}
    )  # more information can be found here: https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjvcamera
