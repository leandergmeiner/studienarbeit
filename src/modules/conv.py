import warnings
from typing import Literal

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        base: nn.Conv2d,
        depth: int,
        num_patches: int,
        method: Literal["center"] | Literal["uniform"] = "center",
        additional_cls_token: bool = True,
        additional_distillation_token: bool = True,
    ):
        super().__init__()

        # FIXME: This is a bit wack
        if depth % 2 == 0 and method == "center":
            depth += 1
            warnings.warn(
                f"{method=} is used and depth is even, therefore a center can not be determined.\n Using {depth=}"
            )

        self.depth = depth

        conv = _conv2d_to_conv3d(base, depth, depth_stride=depth, method=method)

        self.patch_embedding = nn.Sequential(
            Rearrange("b t c h w -> b c t h w"),
            conv,
            Rearrange("b c t h w -> b t (h w) c"),
            nn.Linear(conv.out_channels, conv.out_channels)
        )

        self.states_cls_token = nn.Parameter(torch.zeros(1, conv.out_channels))
        self.states_distillation_token = nn.Parameter(torch.zeros(1, conv.out_channels))

        self.additional_cls_token = additional_cls_token
        self.additional_distillation_token = additional_distillation_token
        num_additional_tokens = int(additional_cls_token) + int(
            additional_distillation_token
        )
        
        self.states_position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + num_additional_tokens, conv.out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        embeddings: Tensor = self.patch_embedding(x)
        b, t = embeddings.shape[:2]

        tokens = [embeddings]

        if self.additional_cls_token:
            cls_tokens = self.states_cls_token.expand(b, t, -1, -1)
            tokens.insert(0, cls_tokens)

        if self.additional_distillation_token:
            distillation_tokens = self.states_distillation_token.expand(b, t, -1, -1)
            tokens.insert(1, distillation_tokens)

        embeddings = torch.cat(tuple(tokens), dim=-2)
        embeddings += self.states_position_embeddings
        return embeddings


# Incomplete conversion from conv2d to conv3d (e.g. missing padding)
# Uses central frame initialisation
def _conv2d_to_conv3d(
    conv2d: nn.Conv2d,
    depth: int,
    depth_stride: int = 1,
    method: Literal["center"] | Literal["uniform"] = "center",
) -> nn.Conv3d:
    h, w = conv2d.weight.shape[2:]
    depth_stride = (depth_stride, *conv2d.stride)

    conv3d = nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        (depth, h, w),
        depth_stride,
    )

    if method == "center":
        weight = rearrange(conv2d.weight, "o i h w -> o i () h w")
        weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, depth // 2, depth // 2))
    elif method == "uniform":
        weight = repeat(conv2d.weight.clone(), "o i h w -> o i d h w", d=depth).clone()
        weight /= depth
    else:
        raise ValueError(method)

    conv3d.weight = nn.Parameter(weight)
    conv3d.bias = conv2d.bias

    return conv3d
