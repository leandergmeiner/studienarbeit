import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import  SubprocVecEnv, make_vec_env
from vizdoom_env import create_vizdoom_env  
import os 
import time

# environment for each map
def make_env(map_name):
    """
    Utility function for creating a single environment.
    :param map_name: Name of the map configuration.
    :return: Callable function to create the environment.
    """
    return lambda: create_vizdoom_env(map=map_name)  # Ensure your `create_vizdoom_env` handles the `map` argument

if __name__ == '__main__':
    model_path = "ppo_vizdoom_model"
    maps = ["scenarios/shooting/basic.cfg", "scenarios/shooting/basic.cfg", "scenarios/shooting/basic.cfg", "scenarios/shooting/basic.cfg"]
    # envs = SubprocVecEnv([make_env(map_name) for map_name in maps])
    envs = make_vec_env(lambda: create_vizdoom_env(), n_envs=1)  # Pass a callable
    model = PPO.load(model_path,envs, verbose=1,tensorboard_log="./ppo_vizdoom_tensorboard/" )
    # Optionally, evaluate the trained model
    obs = envs.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = envs.step(action)
        # Check if any of the environments is done (any() for array-like)
        if dones.any():
            envs.reset()
        time.sleep(0.1)

    envs.close()