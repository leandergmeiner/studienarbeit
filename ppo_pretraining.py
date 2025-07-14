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
    # Use SubprocVecEnv to handle multiple parallel environments
    envs = make_vec_env(lambda: create_vizdoom_env(), n_envs=12)  # Pass a callable
    # envs = DummyVecEnv([make_env(map_name) for map_name in maps])

    print(os.path.exists(model_path))

    if os.path.exists(model_path):
        model = PPO.load(model_path, envs, verbose=1,tensorboard_log="./ppo_vizdoom_tensorboard/", device="cuda")
        model.set_env(env=envs)
    else:
        model = PPO("CnnPolicy",
            envs,
            n_steps=2048,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1,
            tensorboard_log="./ppo_vizdoom_tensorboard/",
            device="cuda")
    
    timesteps= 1000000
    model.learn(timesteps,tb_log_name=f"PPO")
    model.save(f"{model_path}")
    envs.close()
