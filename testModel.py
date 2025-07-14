import gymnasium
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from vizdoom_env import create_vizdoom_env  
import os 
import time

# Environment setup
def make_env(map_name=None):
    return lambda: create_vizdoom_env()

if __name__ == '__main__':
    model_path = "archive/tc_lvl5_1.zip"
    envs = make_vec_env(lambda: create_vizdoom_env(), n_envs=1)
    model = PPO.load(model_path, env=envs, verbose=1)

    obs = envs.reset()

    # Get initial grayscale frame and convert to BGR for video
    frame = envs.envs[0].render()
    frame_shape = (frame.shape[1], frame.shape[0])  # (width, height)

    video_filename = "test_run.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, 30.0, frame_shape)

    for _ in range(4000):
        action, _ = model.predict(obs)
        obs, rewards, dones, _ = envs.step(action)

        frame = envs.envs[0].render()

        # Convert grayscale to BGR so OpenCV can write it
        out.write(frame)

        if dones.any():
            obs = envs.reset()

        time.sleep(0.005)

    out.release()
    envs.close()
    print(f"Video saved as {video_filename}")
