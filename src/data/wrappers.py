import warnings
from collections import defaultdict
from typing import Callable, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np
import vizdoom as vzd
import vizdoom.gymnasium_wrapper  # noqa: F401
from gymnasium.utils import EzPickle
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv

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

        self.unwrapped.observation_space["gamevariables"] = self.observation_space[
            "gamevariables"
        ]
        self.unwrapped.num_game_variables = self.num_game_variables


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
            dtype=np.float32,
        )

        self.unwrapped.observation_space["gamevariables"] = self.observation_space[
            "gamevariables"
        ]
        self.unwrapped.num_game_variables = self.num_game_variables

        self.old_state = None


class VizdoomEnvFromGame(VizdoomEnv):
    def __init__(
        self,
        game: vzd.DoomGame,
        frame_skip: int = 1,
        max_buttons_pressed: int = 1,
        render_mode: vzd.ScreenFormat | None = None,
    ):
        EzPickle.__init__(self, game, max_buttons_pressed, frame_skip, render_mode)

        self.game = game
        self.game.close()

        self.game.set_window_visible(False)

        self.game.add_available_game_variable(vzd.GameVariable.HEALTH)

        self.max_buttons_pressed = max_buttons_pressed
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        screen_format = self.game.get_screen_format()
        if (
            screen_format != vzd.ScreenFormat.RGB24
            and screen_format != vzd.ScreenFormat.GRAY8
        ):
            warnings.warn(
                f"Detected screen format {screen_format.name}. Only RGB24 and GRAY8 are supported in the Gymnasium"
                f" wrapper. Forcing RGB24."
            )
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        self.state = None
        self.clock = None
        self.window_surface = None
        self.isopen = True
        self.channels = 3
        if screen_format == vzd.ScreenFormat.GRAY8:
            self.channels = 1

        self.depth = self.game.is_depth_buffer_enabled()
        self.labels = self.game.is_labels_buffer_enabled()
        self.automap = self.game.is_automap_buffer_enabled()

        # parse buttons defined by config file
        self._VizdoomEnv__parse_available_buttons()

        # check for valid max_buttons_pressed
        if max_buttons_pressed > self.num_binary_buttons > 0:
            warnings.warn(
                f"max_buttons_pressed={max_buttons_pressed} "
                f"> number of binary buttons defined={self.num_binary_buttons}. "
                f"Clipping max_buttons_pressed to {self.num_binary_buttons}."
            )
            max_buttons_pressed = self.num_binary_buttons
        elif max_buttons_pressed < 0:
            raise RuntimeError(
                f"max_buttons_pressed={max_buttons_pressed} < 0. Should be >= 0. "
            )

        # specify action space(s)
        self.max_buttons_pressed = max_buttons_pressed
        self.action_space = self._VizdoomEnv__get_action_space()

        # specify observation space(s)
        self.observation_space = self._VizdoomEnv__get_observation_space()

        # self.game.add_game_args("+viz_nocheat 1")
        # self.game.add_game_args("+viz_respawn_delay 10")

        # self.game.set_console_enabled(True)

        # self.game.add_game_args("+vid_forcesurface 1")
        self.game.init()
