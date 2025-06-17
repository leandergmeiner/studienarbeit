# %%
from typing import Callable, NamedTuple

import torch
import torchrl.envs
import vizdoom as vzd
from einops import rearrange
from torchrl.envs import (
    Compose,
    RenameTransform,
    Resize,
    TargetReturn,
    ToTensorImage,
    UnaryTransform,
)

import src.data._gym_envs  # noqa: F401
from src.data.common import DEFAULT_REWARD_FUNCS, DOOM_BUTTONS


# %%
def get_game_variables_mask(
    wanted_variables: list[vzd.GameVariable],
    available_game_variables: list[vzd.GameVariable] = DEFAULT_REWARD_FUNCS.keys(),
) -> torch.BoolTensor:
    available_game_variables = torch.tensor(
        list(map(lambda v: v.value, available_game_variables))
    )
    wanted_variables = torch.tensor(list(map(lambda v: v.value, wanted_variables)))

    indices = [
        torch.nonzero(wanted == available_game_variables)[0]
        for wanted in wanted_variables
    ]

    return torch.tensor(indices)


def get_action_up_projection(
    available_actions: list[vzd.Button],
    available_buttons: list[vzd.Button] = DOOM_BUTTONS,
):
    available_actions = torch.tensor(list(map(lambda a: a.value, available_actions)))
    available_buttons = torch.tensor(list(map(lambda b: b.value, available_buttons)))

    indices = [
        torch.nonzero(action == available_buttons) for action in available_actions
    ]
    return torch.tensor(indices)


def standard_offline_env_make_transforms():
    return (
        torchrl.envs.RenameTransform(in_keys=["screen"], out_keys=["pixels"]),
        torchrl.envs.ToTensorImage(from_int=False, dtype=torch.uint8),
        torchrl.envs.RenameTransform(
            in_keys=["pixels"], out_keys=[("original", "pixels")], create_copy=True
        ),
    )


class EnvWithTransforms(NamedTuple):
    base_env: str
    make_transforms: Callable[[], list[torchrl.envs.Transform]]


ARNOLD_DTC_ENV_AVAILABLE_ACTIONS = [
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,
]


# Warning: The wanted variables need to match the order of the available game variables
def arnold_env_make_transforms(
    target_return: float | None = None,
    frame_skip: int = 2,
    game_variables=[
        vzd.GameVariable.HEALTH,
        vzd.GameVariable.SELECTED_WEAPON_AMMO,
    ],
):
    # TODO: Solve this more elegantly using categorical actions specs in the env.
    game_variables_mask = get_game_variables_mask(game_variables)
    # action_projection = get_action_up_projection(ARNOLD_DTD_ENV_AVAILABLE_ACTIONS)
    t = [
        *standard_offline_env_make_transforms(),
        torchrl.envs.UnaryTransform(
            in_keys=["gamevariables"],
            out_keys=["gamevariables"],
            fn=lambda t: t[game_variables_mask],
        ),
        torchrl.envs.Resize(60, 108, in_keys=["pixels"]),
        torchrl.envs.UnsqueezeTransform(dim=-4, in_keys=["pixels"]),
        torchrl.envs.CatFrames(N=4, dim=-4, in_keys=["pixels"]),  # hist_size=4
        torchrl.envs.FrameSkipTransform(
            frame_skip
        ),  # from Arnold DefendTheCenter config
    ]

    if target_return is not None:
        t.append(TargetReturn(target_return))

    return t


# ENV_TRANSFORMS = {
#     "sa/ArnoldDefendCenter-v0": EnvWithTransforms(
#         base_env="sa/ArnoldDefendCenter-v0", make_transforms=arnold_env_make_transforms
#     ),
#     "sa/ArnoldHealthGathering-v0": EnvWithTransforms(
#         base_env="sa/ArnoldHealthGathering-v0",
#         make_transforms=partial(
#             arnold_env_make_transforms, game_variables=[vzd.GameVariable.HEALTH]
#         ),
#     ),
#     # TODO:
#     "sa/ArnoldDeathmatch-v0": EnvWithTransforms(
#         base_env="sa/ArnoldDeathmatch-v0", make_transforms=arnold_env_make_transforms
#     ),
#     "sa/ArnoldShotgun-v0": EnvWithTransforms(
#         base_env="sa/ArnoldShotgun-v0",
#         make_transforms=partial(arnold_env_make_transforms, frame_skip=3),
#     ),
# }


def arnold_dataset_make_transforms(
    collector_out_key: str,
    observation_shape: tuple[int, int] = (224, 224),
    reward_key=("next", "reward"),
    rtg_key="return_to_go",
    exclude_next_observation: bool = False,
):
    pixels_keys = (
        ["pixels", ("next", "pixels")] if exclude_next_observation else ["pixels"]
    )

    inverse_transforms = [
        torchrl.envs.Reward2GoTransform(in_keys=[reward_key], out_keys=[rtg_key]),
        torchrl.envs.RenameTransform(
            in_keys=[],
            out_keys=[],
            out_keys_inv=[
                ("original", "pixels"),
                ("next", "original", "pixels"),
            ],
            in_keys_inv=["pixels", ("next", "pixels")],
        ),
        torchrl.envs.ExcludeTransform("labels", "labels_buffer", inverse=True),
        torchrl.envs.DTypeCastTransform(
            dtype_in=torch.int64,
            dtype_out=torch.bool,
            in_keys_inv=[collector_out_key],
        ),
    ]

    if exclude_next_observation:
        inverse_transforms.append(
            torchrl.envs.ExcludeTransform(("next", "pixels"), inverse=True)
        )

    forward_transforms = [
        # Forward
        torchrl.envs.ToTensorImage(
            in_keys=pixels_keys,
            out_keys=pixels_keys,
            shape_tolerant=True,
        ),
        torchrl.envs.ExcludeTransform("original", ("next", "original")),
        torchrl.envs.DTypeCastTransform(
            dtype_in=torch.bool,
            dtype_out=torch.float,
            in_keys=[collector_out_key],
        ),
        torchrl.envs.UnaryTransform(
            in_keys=[collector_out_key],
            out_keys=["target_action"],
            fn=lambda td: td.roll(shifts=-1, dims=-2),
        ),
        torchrl.envs.UnaryTransform(
            in_keys=pixels_keys,
            out_keys=pixels_keys,
            fn=lambda pixels: rearrange(pixels, "... h w -> ... w h"),
        ),
        torchrl.envs.Resize(observation_shape, in_keys=pixels_keys),
    ]

    # TODO: This Rename is awkward
    if exclude_next_observation:
        forward_transforms.append(
            torchrl.envs.RenameTransform(
                in_keys=["pixels"],
                out_keys=["observation"],
            ),
        )
    else:
        forward_transforms.append(
            torchrl.envs.RenameTransform(
                in_keys=[
                    "pixels",
                    ("next", "pixels"),
                ],
                out_keys=[
                    "observation",
                    ("next", "observation"),
                ],
            )
        )

    return [Compose(*inverse_transforms), Compose(*forward_transforms)]


# TODO: Add frame_skip transform
def online_env_make_transforms(
    target_return: float | None = None, observation_shape: tuple[int, int] = (224, 224)
):
    t = [
        RenameTransform(in_keys=["screen"], out_keys=["pixels"]),
        ToTensorImage(),
        UnaryTransform(
            in_keys=["pixels"],
            out_keys=["pixels"],
            fn=lambda pixels: rearrange(pixels, "... h w -> ... w h"),
        ),
        Resize(observation_shape, in_keys=["pixels"]),
        RenameTransform(
            in_keys=["pixels"],
            out_keys=["observation"],
        ),
    ]

    if target_return is not None:
        t.append(TargetReturn(target_return))

    return t


def online_dataset_make_transforms(
    collector_out_key: str = "action",
    reward_key=("next", "reward"),
    exclude_next_observation: bool = False,
    rtg_key="return_to_go",
):
    pixels_keys = (
        ["observation", ("next", "observation")] if exclude_next_observation else ["observation"]
    )

    inverse_transforms = [
        torchrl.envs.Reward2GoTransform(in_keys=[reward_key], out_keys=[rtg_key]),
        torchrl.envs.DTypeCastTransform(
            dtype_in=torch.int64,
            dtype_out=torch.bool,
            in_keys_inv=[collector_out_key],
        ),
        torchrl.envs.UnaryTransform(
            in_keys=[],
            out_keys=[],
            out_keys_inv=pixels_keys,
            in_keys_inv=pixels_keys,
            fn=lambda t: t,
            inv_fn=lambda t: (255 * t).to(torch.uint8),
        ),
    ]

    if exclude_next_observation:
        inverse_transforms.append(
            torchrl.envs.ExcludeTransform(("next", "observation"), inverse=True)
        )

    forward_transforms = [
        # Forward
        torchrl.envs.ToTensorImage(in_keys=pixels_keys, shape_tolerant=True),
        torchrl.envs.DTypeCastTransform(
            dtype_in=torch.bool,
            dtype_out=torch.float,
            in_keys=[collector_out_key],
        ),
        torchrl.envs.UnaryTransform(
            in_keys=[collector_out_key],
            out_keys=["target_action"],
            fn=lambda td: td.roll(shifts=-1, dims=-2),
        ),
    ]

    return (
        Compose(*inverse_transforms),
        Compose(*forward_transforms),
    )
