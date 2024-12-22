import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv,make_vec_env
from vizdoom_env import create_vizdoom_env  
import os 


# environment for each map
def make_env(map_name):
    """
    Utility function for creating a single environment.
    :param map_name: Name of the map configuration.
    :return: Callable function to create the environment.
    """
    return lambda: create_vizdoom_env(map=map_name)  # Ensure your `create_vizdoom_env` handles the `map` argument

if __name__ == '__main__':
    model_path = "ppo_vizdoom_model.zip"
    maps = ["scenarios/shooting/basic.cfg", "scenarios/shooting/basic.cfg", "scenarios/shooting/basic.cfg", "scenarios/shooting/basic.cfg"]
    # Use SubprocVecEnv to handle multiple parallel environments
    envs = make_vec_env(lambda: create_vizdoom_env(), n_envs=6)  # Pass a callable
    # envs = DummyVecEnv([make_env(map_name) for map_name in maps])

    print(os.path.exists(model_path))

    if os.path.exists(model_path):
        model = PPO.load(model_path, envs, verbose=1,tensorboard_log="./ppo_vizdoom_tensorboard/", device="cpu")
        model.set_env(env=envs)
    else:
        # Possible changes: Use sgde: ==> Better exploration, smaller episode length ==> faster covergence...r
        model = PPO("CnnPolicy", envs, verbose=1, tensorboard_log="./ppo_vizdoom_tensorboard/", device="cpu")

    timesteps= 200000
    for i in range(2):
        model.learn(timesteps,tb_log_name=f"PPO_{i}")  
        model.save(f"model_path_{i}")
    envs.close()

