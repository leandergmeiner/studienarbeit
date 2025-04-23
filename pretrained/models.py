# %%
from typing import Literal

from logging import getLogger
from pathlib import Path

import torch
from tensordict import TensorDict

from pretrained.arnold.src.args import finalize_args
from pretrained.arnold.src.model import get_model_class, DQNRecurrent, DQNFeedforward
from pretrained.arnold.src.doom.actions import create_action_set
from pretrained.arnold.src.model.bucketed_embedding import BucketedEmbedding

from src.data.env import DOOM_BUTTONS

import vizdoom as vzd

from einops import rearrange

logger = getLogger()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


ArnoldModelType = Literal["dqn_ff", "dqn_rnn"]
ArnoldScenarioType = Literal["defend_the_center", "deathmatch"]

MODEL_PATHS: dict[ArnoldScenarioType, Path] = {
    "defend_the_center": Path("pretrained/arnold/pretrained/defend_the_center.pth"),
    "deathmatch": Path("pretrained/arnold/pretrained/deathmatch.pth"),
}


class ArnoldAgent(torch.nn.Module):
    def __init__(
        self,
        scenario: ArnoldScenarioType = "defend_the_center",
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
            gpu_id=-1,  # We will move the model to the GPU ourselves
            recurrence="",
            speed="off",
            crouch="off",
            use_continuous=False,
        )

        if scenario == "defend_the_center":
            params.update(
                # frame_skip=2, Not needed; Handled by TorchRL Transform
                network_type="dqn_ff",
                action_combinations="turn_lr+attack",
                game_variables=[("health", 101), ("sel_ammo", 301)],
            )

            # if model_type == "dqn_rnn":
            #     params.update(
            #         recurrence="lstm",
            #         n_rec_layers=1,
            #         remember=1,
            #         batch_size=1,
            #     )
        elif scenario == "deathmatch":
            raise ValueError(f"Unknown scenario: {scenario}")
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        finalize_args(params)

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

        network: DQNFeedforward | DQNRecurrent = get_model_class(params.network_type)(
            params
        )
        logger.info(f"Reloading model from {model_path}...")
        reloaded = torch.load(model_path)
        network.module.load_state_dict(reloaded)

        self.params = params
        self.network = network
        self.screen_shape = self.network.screen_shape

        self.pixels_key = ("pixels",)
        self.game_variables_key = ("gamevariables",)
        self.action_key = ("action",)

        self.network.f_eval = self.f_eval

        BucketedEmbedding.forward = ArnoldAgent.bucketed_embedding_forward

        self.last_frames = []

    def forward(self, tensordict: TensorDict):
        # TODO: Solve this more elegantly using categorical actions specs in the env.
        actions = [
            self.doom_actions[int(action_id)]
            for action_id in self.next_action(tensordict)
        ]
        tensordict[self.action_key] = actions
        return tensordict

    # Overrides of functions that don't work out-of-the-box from Arnold

    def next_action(self, last_states: TensorDict):
        scores, pred_features = self.f_eval(last_states)
        if self.params.network_type == "dqn_ff":
            if pred_features is not None:
                assert pred_features.size() == (1, self.network.module.n_features)
                pred_features = pred_features[0]
        else:
            assert self.params.network_type == "dqn_rnn"
            seq_len = 1 if self.params.remember else self.params.hist_size
            assert scores.size() == (1, seq_len, self.network.module.n_actions)
            scores = scores[0, -1]
            if pred_features is not None:
                assert pred_features.size() == (
                    1,
                    seq_len,
                    self.network.module.n_features,
                )
                pred_features = pred_features[0, -1]

        action_ids = scores.data.max(-1)[1]
        self.pred_features = pred_features
        return action_ids

    def f_eval(self, inputs):
        assert inputs.batch_size

        screen_buffer = inputs[self.pixels_key]
        screen_buffer = rearrange(screen_buffer, "... hist c h w -> ... (hist c) h w")

        return self.network.module(
            screen_buffer,
            inputs[self.game_variables_key].swapaxes(0, -1).long(),
        )

    def bucketed_embedding_forward(self, indices):
        return super(BucketedEmbedding, self).forward(
            indices.div(self.bucket_size, rounding_mode="floor")
        )


# %%
