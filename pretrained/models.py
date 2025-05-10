# %%
from typing import Literal

from logging import getLogger
from pathlib import Path

import torch
from tensordict import TensorDict

from pretrained.common import dotdict
from pretrained.arnold.src.args import finalize_args
from pretrained.arnold.src.model import get_model_class
from pretrained.arnold.src.doom.actions import create_action_set
from pretrained.arnold.src.model.bucketed_embedding import BucketedEmbedding
from pretrained.arnold.src.doom.labels import parse_labels_mapping

from src.data.env import DOOM_BUTTONS

import vizdoom as vzd

from einops import rearrange

logger = getLogger()

ArnoldModelType = Literal["dqn_ff", "dqn_rnn"]
ArnoldScenarioType = Literal[
    "defend_the_center", "health_gathering", "shotgun", "deathmatch"
]

MODEL_PATHS: dict[ArnoldScenarioType, Path] = {
    "defend_the_center": Path("pretrained/arnold/pretrained/defend_the_center.pth"),
    "health_gathering": Path("pretrained/arnold/pretrained/health_gathering.pth"),
    "shotgun": Path("pretrained/arnold/pretrained/deathmatch_shotgun.pth"),
    "deathmatch": Path("pretrained/arnold/pretrained/vizdoom_2017_track1.pth"),
}


class ArnoldAgent(torch.nn.Module):
    def __init__(
        self,
        scenario: ArnoldScenarioType = "defend_the_center",
        batch_size: int = 32,
        model_path: Path | None = None,
        available_buttons: list[vzd.Button] | None = None,
    ):
        super().__init__()

        model_path = model_path or MODEL_PATHS[scenario]

        # Network initialization and optional reloading

        # Taken from the command line defaults and run.sh
        params = dotdict(
            use_bn=False,
            clip_delta=1.0,
            variable_dim="32",
            bucket_size="1",
            hidden_dim=512,
            update_freuency=4,
            dropout=0.0,
            optimizer="rmsprop,lr=0.0002",
            use_screen_buffer=True,
            use_depth_buffer=False,
            action_combinations="move_fb+turn_lr+move_lr+attack",
            game_features="",
            labels_mapping="",
            height=60,
            width=108,
            hist_size=4,
            # gpu_id=-1,  # We will move the model to the GPU ourselves
            gpu_id=-1, # TODO FIXME
            speed="off",
            crouch="off",
            use_continuous=False,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            batch_size=batch_size,
            replay_memory_size=1000000,
            start_decay=0,
            stop_decay=1000000,
            gamma=0.99,
            dueling_network=False,
            recurrence="",
        )

        if scenario == "defend_the_center":
            params.update(
                # frame_skip=2, Not needed; Handled by TorchRL Transform
                network_type="dqn_ff",
                action_combinations="turn_lr+attack",
            )
        elif scenario == "shotgun":
            params.update(
                network_type="dqn_rnn",
                recurrence="lstm",
                n_rec_layers=1,
                n_rec_updates=1,
                remember=True,
                hist_size=6,
                # game_features="target,enemy",
                bucket_size="[10, 1]",
                dropout=0.5,
                action_combinations="move_fb+move_lr;turn_lr;attack",
                speed="on",
                crouch="off",
                labels_mapping="0",
            )
        elif scenario == "deathmatch":
            params.update(
                network_type="dqn_rnn",
                recurrence="lstm",
                n_rec_layers=1,
                n_rec_updates=1,
                remember=True,
                hist_size=4,
                # game_features="target,enemy",
                bucket_size="[10, 1]",
                dropout=0.5,
                action_combinations="move_fb+move_lr;turn_lr;attack",
                speed="on",
                crouch="off",
                # labels_mapping="",
            )
        elif scenario == "health_gathering":
            params.update(
                # frame_skip=2, Not needed; Handled by TorchRL Transform
                network_type="dqn_ff",
                action_combinations="move_fb;turn_lr",
                game_variables=[("health", 101)],
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        finalize_args(params)

        if params.label_mapping is not None:
            self.labels_mapping = parse_labels_mapping(params.label_mapping)

        # action_builder = ActionBuilder(params)
        # params = action_builder.params
        # params.action_builder = action_builder

        available_buttons = available_buttons or DOOM_BUTTONS

        self.available_buttons = list(map(lambda b: b.name, available_buttons))

        self.available_actions = create_action_set(params.action_combinations, False)

        self.doom_actions = []
        for sub_actions in self.available_actions:
            doom_action = [
                button in sub_actions for button in self.available_buttons[:-2]
            ]
            doom_action.append(params.speed == "on")
            doom_action.append(params.crouch == "on")
            self.doom_actions.append(doom_action)
        self.n_actions = len(self.available_actions)
        params.n_actions = self.n_actions

        network = get_model_class(params.network_type)(params)
        logger.info(f"Reloading model from {model_path}...")
        reloaded = torch.load(model_path)

        # We use strict=False since we don't want to load the weights for
        # the auxilliary prediction tasks using game_features
        network.module.load_state_dict(reloaded, strict=False)

        self.params = params
        self.network = network
        self.module = self.network.module
        self.screen_shape = self.network.screen_shape

        self.pixels_key = ("pixels",)
        self.labels_buffer_key = ("labels_buffer",)
        self.game_variables_key = ("gamevariables",)
        self.action_key = ("action",)

        self.network.f_eval = self.f_eval

        def bucketed_embedding_forward(self: BucketedEmbedding, indices):
            return super(BucketedEmbedding, self).forward(
                indices.div(self.bucket_size, rounding_mode="floor").to(torch.int64)
            )

        BucketedEmbedding.forward = bucketed_embedding_forward

        self.last_frames = []

    def forward(self, tensordict: TensorDict):
        # Selected weapon ammo
        # FIXME: Don't hard code this. Why does this occurr in the first place
        old_gamevariables = tensordict["gamevariables"]

        gamevariables = old_gamevariables.clone()
        gamevariables[gamevariables < 0] = 0
        tensordict["gamevariables"] = gamevariables

        # TODO: Solve this more elegantly using categorical actions specs in the env.
        actions = [
            self.doom_actions[int(action_id)]
            for action_id in self.next_action(tensordict)
        ]
        tensordict[self.action_key] = actions
        tensordict["gamevariables"] = old_gamevariables
        return tensordict

    # Overrides of functions that don't work out-of-the-box from Arnold

    def next_action(self, last_states: TensorDict):
        scores, pred_features = self.f_eval(last_states)

        if self.params.network_type == "dqn_rnn":
            seq_len = 1 if self.params.remember else self.params.hist_size
            assert scores.size() == (1, seq_len, self.network.module.n_actions)
            scores = scores[:, -1]

        action_ids = scores.data.max(-1)[1]
        self.pred_features = pred_features
        return action_ids

    def f_eval(self, inputs):
        assert inputs.batch_size
        screen_buffer = inputs[self.pixels_key]
        game_variables = inputs[self.game_variables_key]

        if self.params.network_type == "dqn_ff":
            screen_buffer = rearrange(
                screen_buffer, "... hist c h w -> ... (hist c) h w"
            )

            return self.network.module(
                screen_buffer,
                game_variables.swapaxes(0, -1).long(),
            )
        elif self.params.network_type == "dqn_rnn":
            # if we remember the whole sequence, only feed the last frame
            if self.params.remember:
                screen_buffer = screen_buffer[..., -1, :, :, :]

                if self.labels_buffer_key in inputs:
                    screen_buffer = torch.cat(
                        (screen_buffer, inputs[self.labels_buffer_key]), dim=-3
                    )

                output = self.network.module(
                    screen_buffer.view(1, 1, *self.screen_shape),
                    [
                        game_variables[-1:, i].view(1, 1)
                        for i in range(self.params.n_variables)
                    ],
                    prev_state=self.network.prev_state,
                )
                # save the hidden state if we want to remember the whole sequence
                self.network.prev_state = output[-1]
            # otherwise, feed the last `hist_size` ones
            else:
                output = self.network.module(
                    screen_buffer.view(1, self.network.hist_size, *self.screen_shape),
                    [
                        game_variables[:, i]
                        .contiguous()
                        .view(1, self.network.hist_size)
                        for i in range(self.params.n_variables)
                    ],
                    prev_state=self.network.prev_state,
                )

            # do not return the recurrent state
            return output[:-1]


# %%
