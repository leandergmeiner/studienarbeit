from typing import Mapping

import torch
from einops import rearrange, repeat
from torch import Tensor, nn, vmap
from modules.types import TrajectoryModel


# The reason we don't use torchrl.modules.DecisionTransformer
# is that it does not allow us to specifiy a custom transformer decoder backbone
class VideoDT(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patching: nn.Module,
        pos_embedding: nn.Module,
        spatial_transformer: nn.Module,
        temporal_transformer: nn.Module,
        num_spatial_heads: int,
        num_temporal_heads: int,
        dropout: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.patching = patching
        self.pos_embedding = pos_embedding
        self.spatial_transformer = spatial_transformer
        self.temporal_transformer = temporal_transformer

        self.embed_action = nn.LazyLinear(self.hidden_size)
        self.embed_return = nn.LazyLinear(self.hidden_size)

        self.space_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        # self.temporal_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        self.dropout = dropout

        self.temp_norm = nn.LayerNorm(hidden_size)

        self.num_spatial_heads = num_spatial_heads
        self.num_temporal_heads = num_temporal_heads

    def forward(
        self,
        observation: Tensor,  # b;t;c;h;w
        action: Tensor,
        return_to_go: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        patches: Tensor = self.patching(observation)

        b, t = patches.shape[:2]

        cls_space_token = repeat(self.space_token, "() n d -> b t n d", b=b, t=t)
        # cls_temporal_token = repeat(self.temporal_token, "() n d -> b n d", b=b)

        patches = torch.cat([cls_space_token, patches], dim=2)
        patches = rearrange(
            patches, "b t n (h d) -> b (t n) h d", h=self.num_spatial_heads
        )
        patches = self.pos_embedding(patches)
        patches = rearrange(patches, "b (t n) h d -> b t n (h d)", t=t)
        patches = self.dropout(patches)

        patches = rearrange(patches, "b t ... -> (b t) ...")
        states: Tensor = self.spatial_transformer(patches)["last_hidden_state"]
        states = rearrange(states[:, 0], "(b t) ... -> b t ...", b=b, t=t)

        embedded_actions: Tensor = self.embed_action(action)
        embedded_returns: Tensor = self.embed_return(return_to_go)

        embeddings = [embedded_returns, states, embedded_actions]

        x: Tensor = rearrange(
            embeddings, "e b n (h d) -> e b n h d", h=self.num_temporal_heads
        )
        x = vmap(self.pos_embedding)(x)
        x = rearrange(x, "e b n h d -> b (n e) (h d)")

        # We don't need a temporal_token
        # x = torch.cat([cls_temporal_token, x], dim=1)

        # TODO: Is this correct?
        stacked_attention_mask = (
            repeat(
                attention_mask,
                "s1 s2 -> b (s1 e1) (s2 e2)",
                b=b,
                e1=len(embeddings),
                e2=len(embeddings),
            )
            if attention_mask is not None
            else None
        )

        x = self.temp_norm(x)
        x = self.temporal_transformer(
            inputs_embeds=x, attention_mask=stacked_attention_mask
        )

        if isinstance(x, Mapping):
            x = x["last_hidden_state"]

        x = rearrange(x, "b (n e) d -> b e n d", e=len(embeddings))

        return x[:, 1]


class DTActor(nn.Module):
    def __init__(self, model: TrajectoryModel, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = model
        self.action_layer = nn.Linear(hidden_dim, action_dim)

        self.action_layer.apply(lambda x: nn.init.orthogonal_(x.weight.data))

    def forward(
        self,
        observation: Tensor,
        action: Tensor,
        return_to_go: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_state = self.model(
            observation, action, return_to_go, attention_mask=attention_mask
        )
        out = self.action_layer(hidden_state)
        return out
