from tqdm import tqdm
import gymnasium as gym
import numpy as np

EPSIODES = 100

env = gym.make(
    'Ant-v5',
    render_mode='human',
    xml_file='./custom_gym/assets/DroneWorld/skydio_x2/scene.xml',
    forward_reward_weight=0,
    ctrl_cost_weight=0,
    contact_cost_weight=0,
    healthy_reward=0,
    main_body=1,
    healthy_z_range=(0, np.inf),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0,
    frame_skip=1,
    max_episode_steps=1000,
)


print(env.action_space)
for episode in tqdm(range(EPSIODES)):
    observations, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        next_observations, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        observations = next_observations

        env.render()

env.close()