# %%
from typing import Literal, Any

from logging import getLogger
from pathlib import Path

import torch
from tensordict.nn import TensorDictModule, dispatch
from tensordict import TensorDict

from pretrained.arnold.src.args import finalize_args
from pretrained.arnold.src.model import get_model_class, DQNRecurrent, DQNFeedforward
from pretrained.arnold.src.doom.utils import get_n_feature_maps
from pretrained.arnold.src.doom.actions import ActionBuilder
from pretrained.arnold.src.model.bucketed_embedding import BucketedEmbedding

from src.data.env import (
    VizdoomWithRewardWrapper,
    VizdoomSetGameVariables,
    ObservationStackWrapper,
)

import cv2
import vizdoom as vzd
import gymnasium as gym
import numpy as np
import torchvision
import torchrl

from einops import rearrange

logger = getLogger()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


ArnoldModelType = Literal["dqn_ff"] | Literal["dqn_rnn"]
ArnoldScenarioType = Literal["defend_the_center"]


class ArnoldAgent(torch.nn.Module):
    def __init__(
        self,
        scenario: ArnoldScenarioType = "defend_the_center",
        model_path: Path = Path("pretrained/arnold/pretrained/defend_the_center.pth"),
        model_type: ArnoldModelType = "dqn_ff",
    ):
        super().__init__()

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
            network_type=model_type,
        )

        if scenario == "defend_the_center":
            params.update(
                # frame_skip=2, Not needed; Handled by TorchRL Transform
                action_combinations="turn_lr+attack",
                game_variables=[("health", 101), ("sel_ammo", 301)],
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        finalize_args(params)

        action_builder = ActionBuilder(params)
        params = action_builder.params
        params.action_builder = action_builder

        network: DQNFeedforward | DQNRecurrent = get_model_class(model_type)(params)
        logger.info(f"Reloading model from {model_path}...")
        reloaded = torch.load(model_path)
        network.module.load_state_dict(reloaded)

        self.params = params
        self.network = network
        self.screen_shape = self.network.screen_shape

        self.pixels_key = ("observation",)
        self.game_variables_key = ("gamevariables",)
        self.action_key = ("action",)

        self.network.f_eval = self.f_eval

        BucketedEmbedding.forward = ArnoldAgent.bucketed_embedding_forward

        self.last_frames = []

    # @dispatch(source=["observation", "gamevariables"], dest=["action"])
    def forward(self, tensordict: TensorDict):
        # if not tensordict.batch_size:
        #     tensordict = TensorDict(tensordict.unsqueeze(0), batch_size=1)

        old_screen = tensordict[self.pixels_key]
        tensordict[self.pixels_key] = rearrange(
            tensordict[self.pixels_key], "... w h -> ... h w"
        )
        actions = torch.tensor(
            [
                self.params.action_builder.get_action(int(action_id))
                for action_id in self.next_action(tensordict)
            ]
        )
        tensordict[self.action_key] = actions
        tensordict[self.pixels_key] = old_screen

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

        return self.network.module(
            screen_buffer.view(*inputs.batch_size, -1, *self.screen_shape[1:]),
            inputs[self.game_variables_key].swapaxes(0, -1).long(),
        )

    def bucketed_embedding_forward(self, indices):
        return super(BucketedEmbedding, self).forward(
            indices.div(self.bucket_size, rounding_mode="floor")
        )


gym.register(
    "arnold/DefendCenter-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "defend_the_center.cfg"},
    # autoreset=True,
    additional_wrappers=(
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomSetGameVariables.wrapper_spec(
            game_variables=[
                vzd.GameVariable.HEALTH,
                vzd.GameVariable.SELECTED_WEAPON_AMMO,
            ]
        ),
    ),
)

# %%
