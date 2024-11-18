import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from vizdoom_env import create_vizdoom_env  
import os 

model_path = "ppo_vizdoom_model"
#Create multiple parallel vectorized enviroments
envs = make_vec_env(lambda: create_vizdoom_env(), n_envs=2)  # Pass a callable

if os.path.exists(model_path):
    model = PPO.load(model_path, envs, verbose=1,tensorboard_log="./ppo_vizdoom_tensorboard/" )
else:
    model = PPO("CnnPolicy", envs, verbose=1, tensorboard_log="./ppo_vizdoom_tensorboard/")

timesteps= 100000
model.learn(timesteps,tb_log_name="PPO")  
model.save(model_path)
envs.close()
