# %%
from functools import partial
from typing import Callable, NamedTuple

import torch
import torchrl
import vizdoom as vzd

from src.data.common import DEFAULT_REWARD_FUNCS, DOOM_BUTTONS
import src.data._gym_envs  # noqa: F401

# %%
# TODO: Return to go transform
# TODO: Cat frames for transformer online learning


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


def standard_env_transforms():
    return (
        torchrl.envs.RenameTransform(in_keys=["screen"], out_keys=["pixels"]),
        torchrl.envs.ToTensorImage(from_int=False, dtype=torch.uint8),
        torchrl.envs.RenameTransform(
            in_keys=["pixels"], out_keys=[("original", "pixels")], create_copy=True
        ),
        # torchrl.envs.Resize(120, 160, in_keys=[("original", "pixels")]),
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
def _arnold_make_transforms(
    frame_skip: int = 2,
    game_variables=[
        vzd.GameVariable.HEALTH,
        vzd.GameVariable.SELECTED_WEAPON_AMMO,
    ],
):
    # TODO: Solve this more elegantly using categorical actions specs in the env.
    game_variables_mask = get_game_variables_mask(game_variables)
    # action_projection = get_action_up_projection(ARNOLD_DTD_ENV_AVAILABLE_ACTIONS)
    return (
        *standard_env_transforms(),
        torchrl.envs.UnaryTransform(
            in_keys=["gamevariables"],
            out_keys=["gamevariables"],
            fn=lambda t: t[game_variables_mask],
        ),
        # torchrl.envs.UnaryTransform(
        #     in_keys=[],
        #     out_keys=[],
        #     in_keys_inv=["action"],
        #     out_keys_inv=["action"],
        #     fn=lambda t: t,
        #     inv_fn=lambda t: action_projection[t],
        # ),
        torchrl.envs.Resize(60, 108, in_keys=["pixels"]),
        torchrl.envs.UnsqueezeTransform(dim=-4, in_keys=["pixels"]),
        torchrl.envs.CatFrames(N=4, dim=-4, in_keys=["pixels"]),  # hist_size=4
        torchrl.envs.FrameSkipTransform(
            frame_skip
        ),  # from Arnold DefendTheCenter config
    )


# TODO: Health gathering

ENV_TRANSFORMS = {
    "sa/ArnoldDefendCenter-v0": EnvWithTransforms(
        base_env="sa/ArnoldDefendCenter-v0", make_transforms=_arnold_make_transforms
    ),
    "sa/ArnoldHealthGathering-v0": EnvWithTransforms(
        base_env="sa/ArnoldHealthGathering-v0",
        make_transforms=partial(
            _arnold_make_transforms, game_variables=[vzd.GameVariable.HEALTH]
        ),
    ),
    "sa/ArnoldDeathmatch-v0": EnvWithTransforms(
        base_env="sa/ArnoldDeathmatch-v0", make_transforms=_arnold_make_transforms
    ),
    "sa/ArnoldShotgun-v0": EnvWithTransforms(
        base_env="sa/ArnoldShotgun-v0",
        make_transforms=partial(_arnold_make_transforms, frame_skip=3),
    ),
}


# TODO: GymEnv: categorical_action_encoding = True
# TODO: TargetReturn
def make_env(
    env,
    num_workers: int = 1,
    transforms: list | None = None,
    max_seen_rtg: float | None = None,
    **kwargs,
):
    
    def _make_env(env_name: str, transforms_list: list | None):
        transforms_list = transforms_list or []
        if env_name in ENV_TRANSFORMS:
            env_transforms = ENV_TRANSFORMS[env_name]
            env_name = env_transforms.base_env
            transforms_list.extend(env_transforms.make_transforms())
        else:
            transforms_list = standard_env_transforms()

        if max_seen_rtg is not None:
            transforms_list.append(torchrl.envs.TargetReturn(max_seen_rtg))

        # TODO: Solve this more elegantly using categorical actions specs in the env.
        env_name = torchrl.envs.GymEnv(
            env_name,
            # categorical_action_encoding=True,
            **kwargs,
        )
        transform = torchrl.envs.Compose(*transforms_list)
        return torchrl.envs.TransformedEnv(env_name, transform)

    env_creator = torchrl.envs.EnvCreator(
        _make_env, dict(env_name=env, transforms_list=transforms), **kwargs
    )

    if num_workers:
        # TODO FIXME: ParallelEnv
        return torchrl.envs.SerialEnv(num_workers, env_creator)
    else:
        return torchrl.envs.SerialEnv(1, env_creator)



# %%
