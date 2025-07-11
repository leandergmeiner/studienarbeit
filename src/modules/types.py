from typing import Protocol
from torch import Tensor


class TrajectoryModel(Protocol):
    def forward(
        self,
        observation: Tensor,
        action: Tensor,
        return_to_go: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]: ...
