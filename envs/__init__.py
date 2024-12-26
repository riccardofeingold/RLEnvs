from gymnasium.envs.registration import register

register(
    id="envs/GridWorld-v0",
    entry_point="envs.GridWorld:GridWorldEnv"
)