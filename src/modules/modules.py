from typing import Mapping, Optional

import torch
import torchrl
from einops import rearrange, repeat
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torch import Tensor, nn
from torchrl.modules import DecisionTransformerInferenceWrapper

from src.modules.types import TrajectoryModel

# TODO: Image Processor


# The reason we don't use torchrl.modules.DecisionTransformer
# is that it does not allow us to specifiy a custom transformer decoder backbone
class VideoDT(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frame_skip: int,
        spatial_encoder: nn.Module,
        temporal_transformer: nn.Module,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.frame_skip = frame_skip
        self.spatial_encoder = spatial_encoder
        self.temporal_transformer = temporal_transformer

        self.embed_states = nn.LazyLinear(self.hidden_size)
        self.embed_return = nn.LazyLinear(self.hidden_size)
        self.embed_action = nn.LazyLinear(self.hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.embedding_ln = nn.LayerNorm(hidden_size)

    def forward(
        self,
        observation: Tensor,  # b;t;c;h;w
        action: Tensor,
        return_to_go: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        assert observation.shape[-4] % self.frame_skip == 0

        # Reduces observations with frame_skip
        states: Tensor = self.spatial_encoder(observation)

        if isinstance(states, Mapping):
            states = states["last_hidden_state"]

        # Get [CLS] token of the spatial encoder
        states = states[..., 0, :]  # b;t;embedding_dim

        action = action[..., :: self.frame_skip, :]
        return_to_go = return_to_go[..., :: self.frame_skip, :]

        b, t = states.shape[:2]

        embedded_states: Tensor = self.embed_states(states)
        embedded_actions: Tensor = self.embed_action(action)
        embedded_returns: Tensor = self.embed_return(return_to_go)

        # Order matters here
        embeddings = [embedded_returns, embedded_states, embedded_actions]

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs: Tensor = rearrange(embeddings, "e b n d -> b (n e) d")
        stacked_inputs = self.embedding_ln(stacked_inputs)

        stacked_inputs = self.dropout(stacked_inputs)

        if attention_mask is not None:
            stacked_attention_mask = repeat(
                attention_mask,
                "s1 s2 -> b (s1 e1) (s2 e2)",
                b=b,
                e1=len(embeddings),
                e2=len(embeddings),
            )
        else:
            stacked_attention_mask = None

        position_ids = repeat(
            torch.arange(t), "t -> b (t e)", b=b, e=len(embeddings)
        ).to(stacked_inputs.device)

        outputs = self.temporal_transformer(
            inputs_embeds=stacked_inputs,
            position_ids=position_ids,
            attention_mask=stacked_attention_mask,
        )

        if isinstance(outputs, Mapping):
            outputs = outputs["last_hidden_state"]

        outputs = rearrange(outputs, "b (n e) d -> b e n d", e=len(embeddings))

        # Take each aggregated state (index 1) to later predict the next action
        outputs = outputs[:, 1]

        # Since we've "compressed" the actions by only taking every frame_skip-th action,
        # we reverse that action for gradient descent.
        return repeat(outputs, "b n d -> b (n r) d", r=self.frame_skip)


class OnlineDTActor(torch.nn.Module):
    def __init__(self, model: TrajectoryModel, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = model

        self.action_layer_mean = torch.nn.Linear(hidden_dim, action_dim)
        self.action_layer_logstd = torch.nn.Linear(hidden_dim, action_dim)

        self.log_std_min, self.log_std_max = -5.0, 2.0
        self.ln = torch.nn.LayerNorm(hidden_dim)

        # TODO: Is this kind of initialisation necessary?
        # self.action_layer_mean.apply(lambda x: torch.nn.init.orthogonal_(x.weight.data))
        # self.action_layer_logstd.apply(
        #     lambda x: torch.nn.init.orthogonal_(x.weight.data)
        # )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        return_to_go: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_state = self.model(
            observation, action, return_to_go, attention_mask=attention_mask
        )
        hidden_state = self.ln(hidden_state)
        mu = self.action_layer_mean(hidden_state)
        log_std = self.action_layer_logstd(hidden_state)

        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1.0
        )
        std = log_std.exp()

        return mu, std


class DTInferenceWrapper(DecisionTransformerInferenceWrapper):
    def __init__(
        self,
        policy: TensorDictModule,
        *,
        inference_context: int = 5,
        spec: Optional[torchrl.data.TensorSpec] = None,
        device: torch.device | None = None,
    ):
        super().__init__(
            policy, inference_context=inference_context, spec=spec, device=device
        )

    @staticmethod
    def _check_tensor_dims(reward, obs, action):
        if not (reward.shape[:-1] == obs.shape[:-3] == action.shape[:-1]) or not (
            obs.shape[-3] == 3
        ):
            raise ValueError("Mismatched tensor dimensions.")

    def mask_context(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Mask the context of the input sequences."""
        observation = tensordict.get(self.observation_key).clone()
        action = tensordict.get(self.action_key).clone()
        return_to_go = tensordict.get(self.return_to_go_key).clone()
        self._check_tensor_dims(return_to_go, observation, action)

        observation[..., : -self.inference_context, :, :, :] = 0
        action[..., : -(self.inference_context - 1), :] = (
            0  # as we add zeros to the end of the action
        )
        action = torch.cat(
            [
                action[..., 1:, :],
                torch.zeros(
                    *action.shape[:-2], 1, action.shape[-1], device=action.device
                ),
            ],
            dim=-2,
        )
        return_to_go[..., : -self.inference_context, :] = 0

        tensordict.set(self.observation_key, observation)
        tensordict.set(self.action_key, action)
        tensordict.set(self.return_to_go_key, return_to_go)
        return tensordict


class SpatialTransformerEncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        frame_skip: int = 4,
        resolution: tuple[int, int] | None = (224, 224),
    ):
        from inspect import getfullargspec

        from transformers import AutoModel

        from src.modules.conv import PatchEmbedding

        super().__init__()
        model = AutoModel.from_pretrained(
            model_name_or_path, add_pooling_layer=False
        )

        self.encoder = model.encoder

        if "resolution" in getfullargspec(self.encoder.forward).args:
            self.resolution = resolution
        else:
            self.resolution = None

        self.hidden_size = model.encoder.layer[0].layernorm_after.normalized_shape[0]
        base_conv = model.embeddings.patch_embeddings.projection

        self.patching = PatchEmbedding(
            base_conv,
            frame_skip,
            num_patches=model.embeddings.patch_embeddings.num_patches,
            method="uniform",
            additional_distillation_token=not self.resolution,
        )

    def forward(self, observations: torch.Tensor):
        outputs = self.patching(observations)
        stack = []
        for t in outputs.unbind(-4):
            if self.resolution is not None:
                output = self.encoder(t, resolution=self.resolution)
            else:
                output = self.encoder(t)

            stack.append(output["last_hidden_state"])

        return torch.stack(stack, dim=-4)


class SpatialCNNEncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int = 192,
        encoder: torch.nn.Module | None = None,
    ):
        from transformers import AutoModel

        super().__init__()
        self.encoder = encoder or AutoModel.from_pretrained(model_name_or_path)
        self.linear = torch.nn.LazyLinear(output_dim)

    def forward(self, observations: torch.Tensor):
        observations = torch.stack(
            [
                self.encoder(obs)["last_hidden_state"]
                for obs in observations.unbind(dim=-4)
            ],
            dim=-4,
        )
        observations = rearrange(observations, "... d w h -> ... (d w h)")
        observations = self.linear(observations)
        return rearrange(observations, "... d -> ... 1 d")
