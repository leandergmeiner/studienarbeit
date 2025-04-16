import lightning as L
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from modules.types import TrajectoryModel


class DTActor(torch.nn.Module):
    def __init__(self, model: TrajectoryModel, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = model
        self.action_layer = torch.nn.Linear(hidden_dim, action_dim)

        # TODO: Is this kind of initialisation necessary?
        self.action_layer.apply(lambda x: torch.nn.init.orthogonal_(x.weight.data))

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
        return self.action_layer(hidden_state)

class LightningActor(L.LightningModule):
    def __init__(
        self,
        actor: TensorDictModule,
        criterion: torch.nn.Module,
        metrics: dict[str, torch.nn.Module] | None = None,
        actor_out_key: str | None = None,
        out_key: str | None = "action",
        target_key: str = "target",
    ):
        super().__init__()

        self.actor = actor
        self.criterion = criterion

        self.metrics = metrics or {}

        assert len(self.actor.out_keys) > 0
        self.actor_out_key = actor_out_key or self.actor.out_keys[-1]
        self.out_key = out_key
        self.target_key = target_key

        self.criterion = criterion

    def forward(self, batch: TensorDict) -> TensorDict:
        out = self.actor(batch)
        out[self.out_key] = out[self.actor_out_key]

        return out

    def training_step(self, batch: TensorDict):
        out = self.forward(batch)
        
        # TODO: How to go about the loss?
        target_actions = batch[self.target_key]
        # Key must be "loss"
        out["loss"] = self.criterion(out[self.actor_out_key], target_actions)

    @property
    def device(self):
        return self.actor.device
