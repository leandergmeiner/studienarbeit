# %%
from typing import Callable, TypeVar, SupportsFloat, Any, NamedTuple
from collections import defaultdict

import gymnasium as gym
import vizdoom.gymnasium_wrapper  # noqa: F401
import vizdoom as vzd
import numpy as np
import torchrl

import torch
from torchrl.envs import (
    TransformedEnv,
    Compose,
    Resize,
    ToTensorImage,
    ExcludeTransform,
    FrameSkipTransform,
    RenameTransform,
    Transform,
    TargetReturn,
)

from src.data.transforms import SaveOriginalValuesTransform

from functools import partial

# %%
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActionType")


class VizdoomSetGameVariables(gym.Wrapper):
    def __init__(self, env, game_variables: list[vzd.GameVariable]):
        super().__init__(env)

        self.game = env.unwrapped.game
        # Deinit for adding variables
        self.game.close()
        self.game.set_available_game_variables(game_variables)
        self.game.init()

        self.num_game_variables = self.game.get_available_game_variables_size()

        self.observation_space["gamevariables"] = gym.spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            (self.num_game_variables,),
            dtype=self.observation_space["gamevariables"].dtype,
        )


class WithRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        reward_func: Callable[[gym.Env, ObsType, SupportsFloat], SupportsFloat],
    ):
        super().__init__(env)
        self.reward_func = reward_func
        self.old_state = None

    def step(self, action: ActType):
        results = super().step(action)

        observations, reward, *rest = results
        reward = self.reward_func(self, observations, reward)

        return observations, reward, *rest


class VizdoomWithRewardWrapper(WithRewardWrapper):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        reward_funcs: dict[
            vzd.GameVariable, Callable[[SupportsFloat, SupportsFloat], SupportsFloat]
        ],
        replace_game_variables: bool =True,
    ):
        def _reward_func(
            env: VizdoomWithRewardWrapper, observations: ObsType, reward: SupportsFloat
        ) -> SupportsFloat:
            game_variables: dict[vzd.GameVariable, int] = {
                gv: observations["gamevariables"][idx]
                for idx, gv in enumerate(env.game.get_available_game_variables())
            }
            old_game_variables = env.old_state or defaultdict(int)

            for var, func in reward_funcs.items():
                reward += func(game_variables[var], old_game_variables[var])

            env.old_state = game_variables

            return reward

        super().__init__(env, _reward_func)
        
        self.game = env.unwrapped.game
        # Deinit for adding variables
        self.game.close()

        if replace_game_variables:
            self.game.set_available_game_variables(list(reward_funcs.keys()))
        else:
            for var in reward_funcs.keys():
                self.game.add_available_game_variable(var)

        self.game.init()

        self.num_game_variables = self.game.get_available_game_variables_size()

        self.observation_space["gamevariables"] = gym.spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            (self.num_game_variables,),
            dtype=self.observation_space["gamevariables"].dtype,
        )

        self.old_state = None


class ObservationStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_stacks: int):
        super().__init__(env)

        self.num_stacks = num_stacks
        self.last_states = []

        self.observation_space = gym.vector.utils.batch_space(
            self.observation_space, n=num_stacks
        )

    def observation(self, observation: dict[str, Any]):
        self.last_states.append(observation)

        # update most recent states
        if len(self.last_states) == 1:
            self.last_states.extend([self.last_states[0]] * (self.num_stacks - 1))
        else:
            assert len(self.last_states) == self.num_stacks + 1
            del self.last_states[0]

        dict_of_lists = {
            k: [dic[k] for dic in self.last_states] for k in self.last_states[0]
        }

        return dict_of_lists


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

# TODO: Arnold Reward Functions

# %%
CONFIG_FRAME_SKIP = 4

# TODO: Return to go transform
# TODO: Cat frames for transformer online learning

def get_game_variables_mask(
    wanted_variables: list[vzd.GameVariable],
    available_game_variables: list[vzd.GameVariable] = DEFAULT_REWARD_FUNCS.keys(),
) -> torch.BoolTensor:
    return torch.tensor([
        any(wanted == game_variable for wanted in wanted_variables)
        for game_variable in available_game_variables
    ])


def standard_env_transforms():
    return [
        # Action transformation
        FrameSkipTransform(CONFIG_FRAME_SKIP),
        # Observation transformation
        ToTensorImage(in_keys=["screen"], out_keys=["pixels"]),
        ExcludeTransform("screen"),
        Resize(120, 160),
        RenameTransform(in_keys=["pixels"], out_keys=["observation"]),
    ]


class EnvWithTransforms(NamedTuple):
    base_env: str
    make_transforms: Callable[[], list[Transform]]


# Warning: The wanted variables need to match the order of the available game variables
ARNOLD_ENV_WANTED_GAME_VARIABLES = [
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.SELECTED_WEAPON_AMMO,
]

def _arnold_make_transforms():
    game_variables_mask = get_game_variables_mask(ARNOLD_ENV_WANTED_GAME_VARIABLES)
    return [
            *standard_env_transforms(),
            # SaveOriginalValuesTransform(),  # Save original data for later training
            torchrl.envs.UnaryTransform(
                in_keys=["gamevariables"],
                out_keys=["gamevariables"],
                fn=lambda t: t[game_variables_mask],
                inv_fn=lambda t: t,
            ),
            torchrl.envs.Resize((108, 60), in_keys=["observation"]),
            torchrl.envs.UnsqueezeTransform(dim=-4, in_keys=["observation"]),
            torchrl.envs.CatFrames(N=4, dim=-4, in_keys=["observation"]),
        ]

ENV_TRANSFORMS = {
    "arnold/DefendCenter-v0": EnvWithTransforms(
        base_env="sa/DefendCenter-v0",
        make_transforms=_arnold_make_transforms
    ),
}


# TODO: TargetReturn
def make_env(
    env,
    num_workers: int | None = None,
    transforms=[],
    **kwargs,
):

    def _make_env(env_name: str):
        if env_name in ENV_TRANSFORMS:
            env_transforms = ENV_TRANSFORMS[env_name]
            env_name = env_transforms.base_env
            transforms = env_transforms.make_transforms()

        env_name = torchrl.envs.GymEnv(env_name, **kwargs)
        transform = Compose(*transforms)
        return TransformedEnv(env_name, transform)
    
    env_creator = torchrl.envs.EnvCreator(_make_env, dict(env_name=env))
    
    if num_workers is not None:
        return torchrl.envs.SerialEnv(num_workers, env_creator)
    else:
        return _make_env(env)
        
# %%
