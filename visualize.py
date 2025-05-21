# %%
import torch
from einops import rearrange
from torchrl.collectors.utils import split_trajectories
from torchrl.envs import (
    CatFrames,
    Compose,
    ExcludeTransform,
    GymEnv,
    RenameTransform,
    Resize,
    TargetReturn,
    ToTensorImage,
    TransformedEnv,
    UnaryTransform,
    UnsqueezeTransform,
    step_mdp,
)
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger

from src.data.dataset import DoomStreamingDataModule
from src.modules import LightningSequenceActor

# %%
method = "cnn"

target = 300
steps = 400
device = "cuda:0"
logger = CSVLogger(exp_name="test", log_dir="test_dir", video_format="mp4")
video_recorder = VideoRecorder(
    logger=logger, in_keys=["saved_screen"], tag=f"{method}_video", make_grid=False
)

t = Compose(
    RenameTransform(in_keys=["screen"], out_keys=["pixels"], create_copy=True),
    # ExcludeTransform("screen"),
    ToTensorImage(),
    TargetReturn(target, in_keys=[("next", "reward")], out_keys=[("next", "reward")]),
    UnsqueezeTransform(-4, in_keys=["pixels"]),
    UnsqueezeTransform(-2, in_keys=["action", ("next", "reward")]),
    CatFrames(N=64, dim=-4, in_keys=["pixels"], padding="constant"),
    CatFrames(N=64, dim=-2, in_keys=["action", ("next", "reward")], padding="constant"),
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
    ExcludeTransform("original"),
)


env = TransformedEnv(
    GymEnv("sa/ArnoldDefendCenter-v0"),
    Compose(
        ToTensorImage(
            from_int=False,
            unsqueeze=True,
            dtype=torch.uint8,
            in_keys=["screen"],
            out_keys=["saved_screen"],
            shape_tolerant=True,
        ),
        Resize(224, 224, in_keys=["saved_screen"], out_keys=["saved_screen"]),
        video_recorder,
    ),
)

model = LightningSequenceActor.load_from_checkpoint(
    f"models/{method}/last.ckpt",
    strict=False,
    model=LightningSequenceActor.default_model(
        method=method,
        frame_skip=DoomStreamingDataModule.FRAME_SKIP,
        num_actions=DoomStreamingDataModule.NUM_ACTIONS,
        inference_context=64,
    ),
    rtg_key=("next", "reward"),
)

model = model.to(device)

model.on_predict_start()
model.eval()


# %%

# We're simulating a rollout here

print("Start recording")
input_td = env.reset()
action = env.rand_action()


def next_action(tensordict):
    return action


for i in range(steps):
    rollout_td = env.rollout(
        policy=next_action,
        max_steps=DoomStreamingDataModule.FRAME_SKIP,
        break_when_any_done=False,
        auto_reset=False,
        tensordict=input_td,
    )

    input_td = step_mdp(
        rollout_td[..., -1],
    )

    transformed_td = rollout_td.clone()
    transformed_td = split_trajectories(transformed_td)
    del transformed_td["traj_ids"]
    del transformed_td["mask"]

    transformed_td = transformed_td[:, -1]
    # transformed_td = transformed_td.squeeze(dim=1)
    transformed_td = t._call(transformed_td)

    # CatFrames fills the tensordict the other way around as we need it
    transformed_td["observation"] = transformed_td["observation"].flip(dims=(1,))
    transformed_td["action"] = transformed_td["action"].flip(dims=(1,))
    transformed_td[("next", "reward")] = transformed_td[("next", "reward")].flip(
        dims=(1,)
    )

    del transformed_td[("next", "screen")]
    transformed_td = transformed_td.to(device)

    action = model(transformed_td)
    action.batch_size = ()
    action = action.flatten(0, 1)
    action = action.cpu()

    # transformed_td["observation"] = (transformed_td["observation"] * 255).to(
    #     torch.uint8
    # )
    # video_recorder.forward(transformed_td)

    print(f"Step: {i}")

video_recorder.dump()
print("Done")

# %%
