from gymnasium.envs.registration import register

register(id="GridWorld-v0", entry_point="custom_gym.envs.GridWorld:GridWorldEnv")

register(id="DroneWorld-v0", entry_point="custom_gym.envs.DroneWorld:DroneWorldEnv")
