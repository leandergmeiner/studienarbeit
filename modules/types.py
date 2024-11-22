from typing import Protocol, Literal
from torch import Tensor, nn


class TrajectoryModel(Protocol):
    def __init__(
        self,
        hidden_size: int,
        patching: nn.Conv3d,
        spatial_pos_embedding: nn.Module,
        spatial_transformer: nn.Module,
        temporal_pos_embedding: nn.Module,
        temporal_transformer: nn.Module,
        dropout: nn.Module = nn.Identity(),
    ): ...

    def forward(
        self,
        frames: Tensor,
        actions: Tensor,
        returns_to_go: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


class PatchEmbeddingModel(Protocol):
    def __init__(
        self,
        base: nn.Conv2d,
        depth: int,
        initialisation_method: Literal["center"] | Literal["uniform"] = "center",
    ): ...

    def forward(self, x: Tensor) -> Tensor: ...
