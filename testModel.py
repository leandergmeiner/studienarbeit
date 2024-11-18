import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from vizdoom_env import create_vizdoom_env  
import os 
import time

model_path = "ppo_vizdoom_model"
envs = make_vec_env(lambda: create_vizdoom_env(), n_envs=4)  # Pass a callable
model = PPO.load(model_path, envs, verbose=1,tensorboard_log="./ppo_vizdoom_tensorboard/" )
# Optionally, evaluate the trained model
obs = envs.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = envs.step(action)
    # Check if any of the environments is done (any() for array-like)
    if dones.any():
        envs.reset()

envs.close()