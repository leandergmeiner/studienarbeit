import itertools
from copy import deepcopy

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd

import pretrained.arnold.src.doom.game
from pretrained.arnold.src.doom.game import (
    Game as ArnoldGame,
)
from pretrained.arnold.src.doom.labels import get_label_type_id, parse_labels_mapping
from pretrained.common import dotdict
from src.data.wrappers import VizdoomEnvFromGame

# Needs to be redefined, since the executed file changes and therefore the path determination
# of Arnold is wrong.
pretrained.arnold.src.doom.game.RESOURCES_DIR = "pretrained/arnold/resources"  # noqa: F811


def vizdoom_to_arnold_game_edit_fn(
    env: gym.Wrapper,
    game: vzd.DoomGame,
    binary_buttons: list[vzd.Button],
    *,
    set_render_options: bool = False,
):
    game.set_available_buttons(binary_buttons)

    # TODO: Work with composite spaces
    # env.unwrapped.action_space = env.unwrapped._VizdoomEnv__get_action_space()

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

    game.set_labels_buffer_enabled(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_400X225)

    env.unwrapped.observation_space = env.unwrapped._VizdoomEnv__get_observation_space()


class VizdoomArnoldWithLabelsBuffer(gym.Wrapper):
    def __init__(
        self,
        env,
        width: int | None = None,
        height: int | None = None,
        labels_mapping: str | None = None,
    ):
        super().__init__(env)

        self.width = width
        self.height = height

        if labels_mapping is not None:
            self.labels_mapping = parse_labels_mapping(labels_mapping)
        else:
            self.labels_mapping = None

        if self.labels_mapping is not None:
            self.observation_space = deepcopy(self.observation_space)
            self.observation_space["labels_buffer"] = gym.spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                # TODO: n_feature_maps value is hardcoded for now
                # I don't know yet how to get the value here
                (1, self.height, self.width),
            )

    def step(self, action):
        obs, *rest = super().step(action)

        if self.labels_mapping is not None:
            obs["labels_buffer"] = self.get_labels_buffer()

        return obs, *rest

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if self.labels_mapping is not None:
            obs["labels_buffer"] = self.get_labels_buffer()

        return obs, info

    # Wtf is this function from Arnold???
    def get_labels_buffer(self):
        # split all object labels accross different feature maps
        # enemies / health items / weapons / ammo

        # # naive approach
        # _labels_buffer = np.zeros((max(game.labels_mapping) + 1,)
        #                           + init_shape, dtype=np.uint8)
        # for label in labels:
        #     type_id = get_label_type_id(label)
        #     if type_id is not None:
        #         type_id = game.labels_mapping[type_id]
        #         _labels_buffer[type_id, labels_buffer == label.value] = 255

        # create 4 feature maps, where each value is equal to 255 if the
        # associated pixel is an object of a specific type, 0 otherwise

        labels = self.unwrapped.state.labels
        labels_buffer = self.unwrapped.state.labels_buffer
        init_shape = self.observation_space["screen"].shape[:2]

        _mapping = np.zeros((256,), dtype=np.uint8)
        for label in labels:
            type_id = get_label_type_id(label)
            if type_id is not None:
                _mapping[label.value] = type_id + 1
        # -x is faster than x * 255 and is equivalent for uint8
        __labels_buffer = -(
            _mapping[labels_buffer] == np.arange(1, 5)[:, None, None]
        ).astype(np.uint8)

        n_feature_maps = max(x for x in self.labels_mapping if x is not None) + 1
        if n_feature_maps == 4:
            _labels_buffer = __labels_buffer
        else:
            _labels_buffer = np.zeros((n_feature_maps,) + init_shape, dtype=np.uint8)

            for i in range(4):
                j = self.labels_mapping[i]
                if j is not None:
                    _labels_buffer[j] += __labels_buffer[i]
        # resize
        if init_shape != (self.height, self.width):
            _labels_buffer = np.concatenate(
                [
                    cv2.resize(
                        _labels_buffer[i],
                        (self.width, self.height),
                        interpolation=cv2.INTER_AREA,
                    ).reshape(1, self.height, self.width)
                    for i in range(_labels_buffer.shape[0])
                ],
                axis=0,
            )
        assert _labels_buffer.shape == (
            n_feature_maps,
            self.height,
            self.width,
        )

        return _labels_buffer.astype(np.float32)


class VizdoomArnoldEnv(VizdoomEnvFromGame):
    def __init__(
        self,
        scenario: str,
        frame_skip=1,
        max_buttons_pressed=1,
        render_mode=None,
        **kwargs,
    ):
        params = dotdict(kwargs)
        self.params = params

        arnold_game: ArnoldGame = self._get_arnold_game(scenario, dotdict(kwargs))
        arnold_game.start(params.map_id)

        if params.n_bots:
            arnold_game.init_bots_health(100)

        game = arnold_game.game

        super().__init__(game, frame_skip, max_buttons_pressed, render_mode)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if self.params.use_scripted_marines and self.params.n_bots > 0:
            self.game.send_game_command(
                "pukename set_value always 2 %i" % self.params.n_bots
            )
        else:
            self.game.send_game_command("removebots")
            for i in range(self.params.n_bots):
                self.game.send_game_command("addbot")

        return obs, info

    def _get_arnold_game(self, scenario_type: str, params: dotdict) -> ArnoldGame:
        from pretrained.arnold.src.doom.actions import ActionBuilder

        action_builder = ActionBuilder(params)

        if scenario_type in ("defend_the_center", "health_gathering"):
            scenario = scenario_type
        elif scenario_type == "deathmatch":
            scenario = params.wad
        else:
            raise ValueError(f"Unknown scenario: {scenario_type}")

        return ArnoldGame(
            scenario=scenario,
            action_builder=action_builder,
            score_variable="FRAGCOUNT",
            freedoom=params.freedoom,
            use_screen_buffer=params.use_screen_buffer,
            use_depth_buffer=params.use_depth_buffer,
            labels_mapping="",
            game_features=params.game_features,
            mode="PLAYER",
            player_rank=params.player_rank,
            players_per_game=params.players_per_game,
            render_hud=params.render_hud,
            render_crosshair=params.render_crosshair,
            render_weapon=params.render_weapon,
            freelook=params.freelook,
            n_bots=params.n_bots,
            use_scripted_marines=params.use_scripted_marines,
            visible=False,
            screen_format="RGB24",
            respawn_protect=params.respawn_protect,
        )


gym.register(
    "arnold/DefendCenter-v0",
    VizdoomArnoldEnv,
    kwargs=dict(
        scenario="defend_the_center",
        freedoom=True,
        use_screen_buffer=True,
        use_depth_buffer=False,
        game_features="",
        player_rank=0,
        players_per_game=1,
        render_hud=False,
        render_crosshair=True,
        render_weapon=True,
        freelook=False,
        action_combinations="turn_lr+attack",
        map_id=1,
    ),
)
gym.register(
    "arnold/HealthGathering-v0",
    VizdoomArnoldEnv,
    kwargs=dict(
        scenario="health_gathering",
        supreme=True,
        freedoom=True,
        use_screen_buffer=True,
        use_depth_buffer=False,
        game_features="",
        player_rank=0,
        players_per_game=1,
        render_hud=False,
        render_crosshair=False,
        render_weapon=False,
        respawn_protect=False,
        freelook=False,
        action_combinations="move_fb;turn_lr",
        map_id=1,
        n_bots=0,
    ),
)

gym.register(
    "arnold/Shotgun-v0",
    VizdoomArnoldEnv,
    kwargs=dict(
        scenario="deathmatch",
        freedoom=True,
        use_screen_buffer=True,
        use_depth_buffer=False,
        game_features="target,enemy",
        player_rank=0,
        players_per_game=1,
        render_hud=False,
        render_crosshair=True,
        render_weapon=True,
        freelook=False,
        action_combinations="move_fb+move_lr;turn_lr;attack",
        map_id=7,
        wad="deathmatch_shotgun",
        n_bots=2,
        use_scripted_marines=True,
    ),
    additional_wrappers=(
        VizdoomArnoldWithLabelsBuffer.wrapper_spec(
            labels_mapping="0",
            width=108,
            height=60,
        ),
    ),
)

gym.register(
    "arnold/Deathmatch-v0",
    VizdoomArnoldEnv,
    kwargs=dict(
        scenario="deathmatch",
        freedoom=True,
        use_screen_buffer=True,
        use_depth_buffer=False,
        labels_mapping=None,
        game_features="target,enemy",
        player_rank=0,
        players_per_game=1,
        render_hud=False,
        render_crosshair=True,
        render_weapon=True,
        freelook=False,
        action_combinations="move_fb+move_lr;turn_lr;attack",
        map_id=2,
        wad="full_deathmatch",
        n_bots=8,
        init_bots_health=100,
    ),
)
