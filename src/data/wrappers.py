from typing import Callable, TypeVar, SupportsFloat
from collections import defaultdict

import gymnasium as gym
import vizdoom.gymnasium_wrapper  # noqa: F401
import vizdoom as vzd
import numpy as np


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActionType")


class VizdoomEditGameWrapper(gym.Wrapper):
    def __init__(self, env, edit_fn: Callable[[gym.Wrapper, vzd.DoomGame], None]):
        super().__init__(env)

        game = self.env.unwrapped.game
        game.close()
        edit_fn(self, game)
        game.init()


class VizdoomWithBots(gym.Wrapper):
    def __init__(self, env, n_bots=8):
        super().__init__(env)

        self.n_bots = n_bots

        self.game = self.env.unwrapped.game
        # Deinit for adding variables
        self.game.close()
        # Enable the cheat system (so that we can still
        # send commands to the game in self-play mode)
        self.game.add_game_args("+sv_cheats 1")
        self.game.init()

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)

        command = f"pukename set_value always 2 {self.n_bots}"
        self.env.unwrapped.game.send_game_command(command)

        return out


class VizdoomSetGameVariables(gym.Wrapper):
    def __init__(self, env, game_variables: list[vzd.GameVariable]):
        super().__init__(env)

        self.game = self.env.unwrapped.game
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
        # TODO: Extract this into its own wrapper:
        action = len(self.unwrapped.button_map) - action - 1
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
        replace_game_variables: bool = True,
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
