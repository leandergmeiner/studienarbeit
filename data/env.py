# %%
from typing import Callable, TypeVar, SupportsFloat
from collections import defaultdict

import gymnasium as gym
import vizdoom.gymnasium_wrapper  # noqa: F401
import vizdoom as vzd
import numpy as np
import torchrl

from torchrl.envs import (
    TransformedEnv,
    Compose,
    Resize,
    ToTensorImage,
    ExcludeTransform,
    FrameSkipTransform,
    RenameTransform,
    Transform,
    TargetReturn
)

# %%
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActionType")


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

        for var in reward_funcs.keys():
            self.game.add_available_game_variable(var)

        self.game.init()

        self.num_game_variables = self.game.get_available_game_variables_size()

        self.observation_space["gamevariables"] = gym.spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            (self.num_game_variables,),
            dtype=np.float32,
        )

        self.old_state = None

    # def step(self, action):
    #     observations, *rest = super().step(action)
    #     return observations["screen"], *rest

    # def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
    #     observations, info = super().reset(seed=seed, options=options)
    #     return observations["screen"], info


def delta(factor: SupportsFloat):
    def _delta_func(new: SupportsFloat, old: SupportsFloat):
        return factor * (new - old)

    return _delta_func


def leru(f1: SupportsFloat, f2: SupportsFloat):
    def _leru_func(new: SupportsFloat, old: SupportsFloat):
        d = new - old
        return f1 * min(0, d) + f2 * max(0, d)

    return _leru_func


# %%
reward_funcs = {
    vzd.GameVariable.DEATHCOUNT: delta(-5000),
    vzd.GameVariable.KILLCOUNT: delta(1000),
    vzd.GameVariable.ITEMCOUNT: delta(100),
    vzd.GameVariable.ARMOR: delta(10),
    vzd.GameVariable.DAMAGE_TAKEN: delta(10),
    vzd.GameVariable.DAMAGECOUNT: delta(30),
    vzd.GameVariable.SELECTED_WEAPON_AMMO: leru(1, 10),
}

gym.register(
    "sa/DefendLine-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "defend_the_line.cfg"},
    # autoreset=True,
    additional_wrappers=(
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=reward_funcs),
    ),
)


# %%
CONFIG_FRAME_SKIP = 4

# TODO: Return to go transform
# TODO: Cat frames for transformer online learning

def make_env(env, transforms: list[Transform], **kwargs):
    # TODO: TargetReturn

    env = torchrl.envs.GymEnv(env, **kwargs)
    transform = Compose(
        # Action transformation
        FrameSkipTransform(CONFIG_FRAME_SKIP),
        # Observation transformation
        ToTensorImage(in_keys=["screen"], out_keys=["pixels"]),
        ExcludeTransform("screen"),
        Resize(120, 160),
        RenameTransform(in_keys=["pixels"], out_keys=["observation"]),
        **transforms
    )
    env = TransformedEnv(env, transform)
    env.reset()

    return env


# %%
