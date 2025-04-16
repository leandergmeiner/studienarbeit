import copy
import warnings
from functools import partial
from typing import Literal

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, einsum, nn
from torchtune.modules import RotaryPositionalEmbeddings


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        base: nn.Conv2d | None,
        depth: int,
        method: Literal["center"] | Literal["uniform"] = "center",
    ):
        super().__init__()

        # FIXME: This is a bit wack
        if depth % 2 == 0 and method == "center":
            depth += 1
            warnings.warn(
                f"{method=} is used and depth is even, therefore a center can not be determined.\n Using {depth=}"
            )

        self.conv = _conv2d_to_conv3d(base, depth, depth_stride=depth, method=method)

        self.patch_embedding = nn.Sequential(
            Rearrange("b t c h w -> b c t h w"),
            self.conv,
            Rearrange("b c t h w -> b t (h w) c"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.patch_embedding(x)


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
        weight = repeat(conv2d.weight, "o i h w -> o i d h w", d=depth)
        weight /= depth
    else:
        raise ValueError(method)

    conv3d.weight = nn.Parameter(weight)
    conv3d.bias = conv2d.bias

    return conv3d
