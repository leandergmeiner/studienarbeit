import copy
from functools import partial
from typing import Callable, Protocol, Sequence, Mapping

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, einsum, nn
from torchtune.modules import RotaryPositionalEmbeddings

# TODO: Move interfaces into different file


class TrajectoryModel(Protocol):
    def forward(
        self,
        frames: Tensor,
        actions: Tensor,
        returns_to_go: Tensor,
        timesteps: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],  # h;w
        tubelet_dims: tuple[int, int],  # t;s
        embed_dim: int,
        encoder: nn.Module,
        max_ep_len=4096,
        space_dropout=0.0,
        conv_3d: nn.Module | None = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.encoder = encoder

        tubelet_dims_3d = (tubelet_dims[0], tubelet_dims[0], tubelet_dims[1])
        self.patch_embedding = nn.Sequential(
            Rearrange("b t c h w -> b c t h w"),
            conv_3d or nn.LazyConv3d(self.embed_dim, tubelet_dims_3d, tubelet_dims_3d),
            Rearrange("b c t h w -> b t (h w) c"),
        )

        # TODO: Relative Embedding
        h, w = image_size
        patch_size, _ = tubelet_dims
        num_patches = (w // patch_size) * (h // patch_size)

        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_ep_len, num_patches + 1, self.embed_dim)
        )

        self.space_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.embed_norm = nn.LayerNorm(self.embed_dim)

        self.dropout = nn.Dropout(space_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embedding(x)

        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, "() n d -> b t n d", b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        # FIXME:
        # x += self.pos_encoding[:, :, : (n + 1)]
        x = self.dropout(x)

        x = rearrange(x, "b t n d -> (b t) n d")
        print(x.shape)
        x = self.encoder(x)
        x = rearrange(x[:, 0], "(b t) ... -> b t ...", b=b)

        return x


# TODO: Remove timesteps and replace them by learned relative positional encodings (?)
# TODO: Remove embed_dim
class VideoDT(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],  # h;w
        tubelet_dims: tuple[int, int],  # t;s
        hidden_size: int,
        spatial_transformer: nn.Module,
        temporal_transformer: nn.Module,
        max_ep_len=4096,
        space_dropout=0.0,
        embed_frames: nn.Module | None = None,
        conv_3d: nn.Module | None = None,
        predict_action: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.embed_action = nn.LazyLinear(self.hidden_size)
        self.embed_return = nn.LazyLinear(self.hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, self.hidden_size)

        self.embed_frames = embed_frames or SpatialEncoder(
            image_size,
            tubelet_dims,
            self.hidden_size,
            spatial_transformer,
            max_ep_len,
            space_dropout,
            conv_3d,
        )

        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.temporal_transformer = temporal_transformer

        self.embed_norm = nn.LayerNorm(self.hidden_size)

        self.dropout = nn.Dropout(space_dropout)

        self.predict_action = predict_action

    def forward(
        self,
        frames: Tensor,  # b;t;c;h;w
        actions: Tensor,
        returns_to_go: Tensor,
        timesteps: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # TODO: Assert all tensors same batch len and seq len

        state_embeddings: Tensor = self.embed_frames(frames)
        action_embeddings: Tensor = self.embed_action(actions)
        returns_embeddings: Tensor = self.embed_return(returns_to_go)
        time_embeddings: Tensor = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        b, seq_len, _ = state_embeddings.shape
        if mask is None:
            mask = torch.ones((b, seq_len), dtype=state_embeddings.dtype)

        # TODO: Understand this code
        x = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(b, 3 * seq_len, self.hidden_size)
        )

        stacked_attention_mask = (
            torch.stack((mask, mask, mask), dim=1)
            .permute(0, 2, 1)
            .reshape(b, 3 * seq_len)
        )

        x = self.embed_norm(x)

        cls_temporal_tokens = repeat(self.temporal_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        # TODO Use temporal token if temporal transformer is encoder model
        x: Tensor = self.temporal_transformer(x, mask=stacked_attention_mask)

        x = x.reshape(b, seq_len, 3, self.hidden_size).permute(0, 2, 1, 3)

        # TODO: Maybe do predict action/state/return here?

        return x


# Incomplete conversion from conv2d to conv3d (e.g. missing padding)
# Uses central frame initialisation
def conv2d_to_conv3d(conv2d: nn.Conv2d, depth: int, depth_stride: int = 1) -> nn.Conv3d:
    out_channels, in_channels, h, w = conv2d.weight.shape
    depth_stride = (depth_stride, *conv2d.stride)

    conv3d = nn.Conv3d(
        in_channels,
        out_channels,
        (depth, h, w),
        depth_stride,
    )

    weight = rearrange(conv2d.weight, "o i h w -> o i () h w")
    weight = torch.nn.functional.pad(
        weight, (0, 0, 0, 0, depth // 2, depth - depth // 2)
    )

    conv3d.weight = nn.Parameter(weight)
    conv3d.bias = conv2d.bias

    return conv3d
