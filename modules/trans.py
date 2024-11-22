import copy
from functools import partial
from typing import Callable, Protocol, Sequence, Mapping

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, einsum, nn

from torchtune.modules import (
    TransformerDecoder,
    TransformerSelfAttentionLayer,
    MultiHeadAttention,
)


# TODO: Remove timesteps and replace them by learned relative positional encodings (?)
# TODO: Remove embed_dim
class VideoDT(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patching: nn.Conv3d,
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
        frames: Tensor,  # b;t;c;h;w
        actions: Tensor,
        returns_to_go: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        patches: Tensor = self.patching(frames)

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
        # FIXME: This outputs a tuple
        states: Tensor = self.spatial_transformer(patches)["last_hidden_state"]
        states = rearrange(states[:, 0], "(b t) ... -> b t ...", b=b, t=t)

        embedded_actions: Tensor = self.embed_action(actions)
        embedded_returns: Tensor = self.embed_return(returns_to_go)

        embeddings = [embedded_returns, states, embedded_actions]

        # TODO: Is this application of the pos embeddings correct? [pos(e) for e in embeddings]
        # I think not!
        x: Tensor = rearrange(
            embeddings, "e b n (h d) -> b (n e) h d", h=self.num_temporal_heads
        )
        x = self.pos_embedding(x)
        x = rearrange(x, "b n h d -> b n (h d)")
        #  TODO: pos embeddings
        x = self.temp_norm(x)
        # We don't need a temporal_token
        # x = torch.cat([cls_temporal_token, x], dim=1)

        # TODO: Is this correct?
        stacked_attention_mask = (
            repeat(
                mask,
                "s1 s2 -> b (s1 e1) (s2 e2)",
                b=b,
                e1=len(embeddings),
                e2=len(embeddings),
            )
            if mask is not None
            else None
        )

        print(stacked_attention_mask.shape)

        x = self.temporal_transformer(x, mask=stacked_attention_mask)
        x = rearrange(x, "b (n e) d -> b e n d", e=len(embeddings))

        return {"last_hidden_state": x}


def get_simple_temporal_decoder(
    hidden_size: int, num_heads: int, max_seq_len: int, num_layers: int
):
    head_dim = hidden_size // num_heads

    decoder = TransformerDecoder(
        tok_embeddings=torch.nn.Identity(),
        layers=TransformerSelfAttentionLayer(
            MultiHeadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                q_proj=torch.nn.LazyLinear(hidden_size, bias=False),
                k_proj=torch.nn.LazyLinear(hidden_size, bias=False),
                v_proj=torch.nn.LazyLinear(hidden_size, bias=False),
                output_proj=torch.nn.LazyLinear(hidden_size, bias=False),
            ),
            mlp=torch.nn.LazyLinear(hidden_size),
            sa_norm=torch.nn.LayerNorm(hidden_size),
            mlp_norm=torch.nn.LayerNorm(hidden_size),
            sa_scale=lambda attn: attn * (hidden_size**-0.5),
        ),
        num_layers=num_layers,
        max_seq_len=3 * max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=torch.nn.LayerNorm(hidden_size),
        output=torch.nn.LazyLinear(hidden_size),
    )

    return decoder
