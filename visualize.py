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
    ExcludeTransform,
)
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger

from src.data.dataset import DoomStreamingDataModule
from src.modules import LightningDecisionTransformer

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True

model_type = "transformer"

frame_skip = 1
target = 300
steps = 200
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

print("Loading model")

model = LightningDecisionTransformer.load_from_checkpoint(
    f"rsrc/{model_type}/models/last.ckpt",
    model_type=model_type,
    # frame_skip=DoomStreamingDataModule.FRAME_SKIP,
    frame_skip=1,
    num_actions=DoomStreamingDataModule.NUM_ACTIONS,
    inference_context=64,
    rtg_key="target_return",
)

model.eval()
model.to(device)
model._model = torch.compile(model._model)

print("Starting rollout")

with torch.no_grad():
    td = env.rollout(steps, model)
env.transform.dump()

print("Done")
