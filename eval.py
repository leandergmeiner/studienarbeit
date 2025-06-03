import glob
import json
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import torch
from einops import rearrange
from torchrl.envs import (
    Compose,
    ExcludeTransform,
    GymEnv,
    RenameTransform,
    Resize,
    TargetReturn,
    ToTensorImage,
    TransformedEnv,
    UnaryTransform,
    SerialEnv,
)
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger

from src.data.dataset import DoomStreamingDataModule
from src.data.transforms import arnold_env_make_transforms
from src.modules import LightningDecisionTransformer
from pretrained.models import ArnoldAgent

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True

model_type = "transformer"

frame_skip = 1
target = 50
steps = 400
device = "cuda:0"
logger = CSVLogger(exp_name="test", log_dir="test_dir", video_format="mp4")
video_recorder = VideoRecorder(
    logger=logger, in_keys=["saved_screen"], tag=f"{model_type}_video", make_grid=False
)

inference_context = 64

t = Compose(
    RenameTransform(in_keys=["screen"], out_keys=["pixels"], create_copy=True),
    ToTensorImage(),
    UnaryTransform(
        in_keys=["pixels"],
        out_keys=["pixels"],
        fn=lambda pixels: rearrange(pixels, "... h w -> ... w h"),
    ),
    Resize((224, 224), in_keys=["pixels"]),
    RenameTransform(
        in_keys=["pixels"],
        out_keys=["observation"],
    ),
    TargetReturn(target),
    ToTensorImage(in_keys=["screen"], out_keys=["saved_screen"], from_int=False),
    Resize((224, 224), in_keys=["saved_screen"]),
    video_recorder,
    ExcludeTransform("screen", "saved_screen"),
)

env = GymEnv("sa/ArnoldDefendCenter-v0")
env = TransformedEnv(env, t)


def sort_checkpoint_names(file_name: str):
    step = file_name.split("step=")[1]
    step = step.split(".")[0]
    return int(step)


files = f"rsrc/{model_type}/models/*.ckpt"
files = glob.iglob(files)
files = sorted(
    filter(lambda f: not f.endswith("last.ckpt"), files), key=sort_checkpoint_names
)


@torch.no_grad()
def eval_mean_reward_model(policy: Callable | None, env: GymEnv):
    r = []
    for _ in range(3):
        td = env.rollout(steps, policy)
        reward = td[("next", "reward")].cumsum(-2).max()
        r.append(reward)

        if policy is not None and hasattr(policy, "reset"):
            policy.reset()

    return r


mean_rewards = []
for ckpt_file in files:
    print("Evaluating checkpoint", ckpt_file)
    model = LightningDecisionTransformer.load_from_checkpoint(
        ckpt_file,
        model_type=model_type,
        frame_skip=1,
        num_actions=DoomStreamingDataModule.NUM_ACTIONS,
        inference_context=64,
        rtg_key="target_return",
    )

    model.eval()
    model.to(device)
    model._model = torch.compile(model._model)

    mean_rewards.append(eval_mean_reward_model(model, env))

rand_mean_reward = eval_mean_reward_model(None, env)
arnold_mean_reward = eval_mean_reward_model(
    torch.compile(ArnoldAgent("defend_the_center")),
    SerialEnv(
        1,
        lambda: TransformedEnv(
            GymEnv("sa/ArnoldDefendCenter-v0"), Compose(*arnold_env_make_transforms())
        ),
    ),
)


data = dict(
    mean_rewards=mean_rewards,
    rand_mean_reward=rand_mean_reward,
    arnold_mean_reward=arnold_mean_reward,
)

with open("eval_reward.json", "w") as f:
    json.dump(data, f)

print("Done")


def make_graph(
    mean_rewards: list[list[float]] | np.ndarray,
    rand_mean_reward: float,
    arnold_mean_reward: float,
    checkpoint_step: int = 2000,
):
    mean_rewards = np.array(mean_rewards)
    x = np.arange(mean_rewards.shape[0]).repeat(mean_rewards.shape[1]).flatten()
    x *= checkpoint_step
    y = mean_rewards.flatten()

    x = pd.Series(x, name="Checkpoint")
    y = pd.Series(y, name="Achieved reward")

    ax = sns.lineplot(x=x, y=y, label="Unser Modell")
    ax.axhline(
        np.mean(rand_mean_reward),
        0.0,
        1.0,
        color="red",
        linestyle="dashed",
        label="Zufällige Action",
    )
    ax.axhline(
        np.mean(arnold_mean_reward),
        0.0,
        1.0,
        color="green",
        linestyle="dashed",
        label="Datensatz Policy",
    )
    ax.legend(loc="upper left")

    return ax


ax = make_graph(
    mean_rewards=mean_rewards,
    rand_mean_reward=rand_mean_reward,
    arnold_mean_reward=arnold_mean_reward,
)
plt.savefig("rewards.png")
