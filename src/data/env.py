# %%
from typing import Callable, SupportsFloat, NamedTuple

import gymnasium as gym
import vizdoom.gymnasium_wrapper  # noqa: F401
import vizdoom as vzd
import numpy as np
import torchrl

import torch

from functools import partial
import itertools

from src.data.wrappers import VizdoomWithRewardWrapper, VizdoomEditGameWrapper

# %%
DOOM_BUTTONS = [
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.ATTACK,
    vzd.Button.SPEED,
    vzd.Button.CROUCH,
]


def delta(factor: SupportsFloat):
    def _delta_func(new: SupportsFloat, old: SupportsFloat):
        return factor * (new - old)

    return _delta_func


# Leaky ReLU
def leru(f1: SupportsFloat, f2: SupportsFloat):
    def _leru_func(new: SupportsFloat, old: SupportsFloat):
        d = new - old
        return f1 * min(0, d) + f2 * max(0, d)

    return _leru_func


def _doom_game_edit_fn(
    env: gym.Wrapper, game: vzd.DoomGame, *, set_render_options: bool = False
):
    binary_buttons = DOOM_BUTTONS

    game.set_available_buttons(binary_buttons)

    env.unwrapped.num_delta_buttons = 0
    env.unwrapped.num_binary_buttons = len(binary_buttons)
    env.unwrapped.button_map = [
        np.array(list(action))
        for action in itertools.product((0, 1), repeat=env.num_binary_buttons)
        if (env.unwrapped.max_buttons_pressed >= sum(action) >= 0)
    ]
    env.unwrapped.action_space = gym.spaces.Discrete(len(env.button_map))

    if set_render_options:
        # From Arnold Config
        game.set_render_hud(False)  # This helps Arnold a lot!!
        game.set_render_crosshair(True)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.set_render_effects_sprites(False)
        game.set_render_corpses(False)
        game.set_render_screen_flashes(False)


# %%
DEFAULT_REWARD_FUNCS = {
    vzd.GameVariable.DEATHCOUNT: delta(-5000),
    vzd.GameVariable.KILLCOUNT: delta(1000),
    vzd.GameVariable.ITEMCOUNT: delta(100),
    vzd.GameVariable.ARMOR: delta(10),
    vzd.GameVariable.HEALTH: delta(10),
    vzd.GameVariable.DAMAGE_TAKEN: delta(10),
    vzd.GameVariable.DAMAGECOUNT: delta(30),
    vzd.GameVariable.SELECTED_WEAPON_AMMO: leru(1, 10),
}

gym.register(
    "sa/DefendLine-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "defend_the_line.cfg"},
    additional_wrappers=(
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=DEFAULT_REWARD_FUNCS),
    ),
)

gym.register(
    "sa/DefendCenter-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "defend_the_center.cfg"},
    additional_wrappers=(
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=DEFAULT_REWARD_FUNCS),
    ),
)

gym.register(
    "sa/Deathmatch-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "deathmatch.cfg"},
    additional_wrappers=(
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=DEFAULT_REWARD_FUNCS),
    ),
)


gym.register(
    "sa/ArnoldDefendCenter-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "defend_the_center.cfg"},
    additional_wrappers=(
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomEditGameWrapper.wrapper_spec(
            edit_fn=partial(_doom_game_edit_fn, set_render_options=True)
        ),
        VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=DEFAULT_REWARD_FUNCS),
    ),
)


# TODO: Arnold Reward Functions

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
    return [
        torchrl.envs.RenameTransform(in_keys=["screen"], out_keys=["pixels"]),
        torchrl.envs.ToTensorImage(from_int=False, dtype=torch.uint8),
        torchrl.envs.RenameTransform(
            in_keys=["pixels"], out_keys=[("original", "pixels")], create_copy=True
        ),
        # torchrl.envs.Resize(120, 160, in_keys=[("original", "pixels")]),
    ]


class EnvWithTransforms(NamedTuple):
    base_env: str
    make_transforms: Callable[[], list[torchrl.envs.Transform]]


# Warning: The wanted variables need to match the order of the available game variables
ARNOLD_DTD_ENV_WANTED_GAME_VARIABLES = [
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.SELECTED_WEAPON_AMMO,
]

ARNOLD_DTD_ENV_AVAILABLE_ACTIONS = [
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,
]


def _arnold_make_transforms():
    # TODO: Solve this more elegantly using categorical actions specs in the env.
    game_variables_mask = get_game_variables_mask(ARNOLD_DTD_ENV_WANTED_GAME_VARIABLES)
    # action_projection = get_action_up_projection(ARNOLD_DTD_ENV_AVAILABLE_ACTIONS)
    return [
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
        #     inv_fn=lambda t: print(t) or action_projection[t],
        # ),
        torchrl.envs.Resize(60, 108, in_keys=["pixels"]),
        torchrl.envs.UnsqueezeTransform(dim=-4, in_keys=["pixels"]),
        torchrl.envs.CatFrames(N=4, dim=-4, in_keys=["pixels"]),  # hist_size=4
        torchrl.envs.FrameSkipTransform(2),  # from Arnold DefendTheCenter config
    ]


ENV_TRANSFORMS = {
    "sa/ArnoldDefendCenter-v0": EnvWithTransforms(
        base_env="sa/ArnoldDefendCenter-v0", make_transforms=_arnold_make_transforms
    ),
    "sa/ArnoldDeathmatch-v0": EnvWithTransforms(
        base_env="sa/Deathmatch-v0", make_transforms=_arnold_make_transforms
    ),
}


# TODO: GymEnv: categorical_action_encoding = True
# TODO: TargetReturn
def make_env(
    env,
    num_workers: int | None = None,
    transforms=[],
    **kwargs,
):
    def _make_env(env_name: str, transforms_list: list = []):
        if env_name in ENV_TRANSFORMS:
            env_transforms = ENV_TRANSFORMS[env_name]
            env_name = env_transforms.base_env
            transforms_list = env_transforms.make_transforms()

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

    if num_workers is not None and num_workers > 0:
        # TODO FIXME: ParallelEnv
        return torchrl.envs.SerialEnv(num_workers, env_creator)
    else:
        return _make_env(env, transforms_list=transforms)


# %%
