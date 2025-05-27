import torch
from einops import rearrange
from torchrl.envs import (
    Compose,
    GymEnv,
    RenameTransform,
    Resize,
    TargetReturn,
    ToTensorImage,
    TransformedEnv,
    UnaryTransform,
    step_mdp,
)
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger

from src.data.dataset import DoomStreamingDataModule
from src.modules import LightningDecisionTransformer
from tensordict import TensorDict

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True

model_type = "transformer"

frame_skip = DoomStreamingDataModule.FRAME_SKIP
target = 300
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
    # video_recorder,
    # ExcludeTransform("screen"),
)

logger = CSVLogger(exp_name="test", log_dir="test_dir", video_format="mp4")
video_recorder = VideoRecorder(
    logger=logger, in_keys=["saved_screen"], tag=f"{model_type}_video", make_grid=False
)

env = GymEnv("sa/ArnoldDefendCenter-v0")
env = TransformedEnv(env, t)

print("Loading model")

model = LightningDecisionTransformer.load_from_checkpoint(
    f"models/{model_type}/last.ckpt",
    model_type=model_type,
    frame_skip=DoomStreamingDataModule.FRAME_SKIP,
    num_actions=DoomStreamingDataModule.NUM_ACTIONS,
    inference_context=64,
    rtg_key="target_return",
)

model.eval()
model.to(device)

print("Starting rollout")

# We need to do a manual rollout, since env.rollout does not
# respect the FrameSkipTransform, apparently

input_td = env.reset()
action = TensorDict(action=[0])


def next_action(tensordict):
    return action


actions = []
obs = []
for i in range(steps):
    rollout_td = env.rollout(
        policy=next_action,
        max_steps=model.frame_skip,
        break_when_any_done=False,
        auto_reset=False,
        tensordict=input_td,
    )
    input_td = step_mdp(
        rollout_td[..., -1],
    )
    obs.append(rollout_td["saved_screen"])
    action = model(input_td)
    # action["action"] = action["action"].flip(-1).roll(-2)
    # action["action"] = action["action"].flip(-1)
    actions.append(action["action"])

video_recorder.obs = torch.cat(obs).unbind(0)
video_recorder.dump()
print("Done")
