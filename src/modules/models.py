from typing import Mapping

import torch
from einops import rearrange, repeat
from torch import Tensor, nn, vmap


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
        dropout: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.frame_skip = frame_skip
        self.spatial_encoder = spatial_encoder
        self.temporal_transformer = temporal_transformer

        self.embed_states = nn.LazyLinear(self.hidden_size)
        self.embed_return = nn.LazyLinear(self.hidden_size)
        self.embed_action = nn.LazyLinear(self.hidden_size)

        self.dropout = dropout

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
            stacked_attention_mask = (
                repeat(
                    attention_mask,
                    "s1 s2 -> b (s1 e1) (s2 e2)",
                    b=b,
                    e1=len(embeddings),
                    e2=len(embeddings),
                )
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

    def get_states(self, observation: Tensor):
        patches = self.patching(observation)
        patches = self.dropout(patches)

        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        # states: Tensor = vmap(self.spatial_transformer, in_dims=-4)(
        #     patches, resolution=observation.shape[-2:]
        # )
        states: Tensor = vmap(self.spatial_encoder, in_dims=-4)(observation)

        if isinstance(states, Mapping):
            states = states["last_hidden_state"]

        # Get [CLS] token of the spatial encoder
        states = states[..., 0, :]  # b;t;embedding_dim

        return states
