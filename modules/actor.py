import math
from typing import Any

import lightning as L
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from modules.types import TrajectoryModel
from dataloaders.gym import GymnasiumDataloader


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
        out = self.action_layer(hidden_state).softmax(dim=-1)
        return out


class OnlineActor(L.LightningModule):
    def __init__(
        self,
        actor: TensorDictModule,
        criterion: torch.nn.Module,
        metrics: dict[str, torch.nn.Module] | None,
        out_key: str | None = None,
        target_key: str = "target",
        topk: int = -1,
        do_sample: bool = False,
    ):
        super().__init__()

        self.actor = actor
        self.criterion = criterion

        self.metrics = metrics or {}

        assert len(self.actor.out_keys) > 0
        self.out_key = out_key or self.actor.out_keys[-1]
        self.target_key = target_key

        self.criterion = criterion

        self.topk = topk
        self.do_sample = do_sample

    def forward(self, batch: TensorDict) -> TensorDict:
        return self.actor(batch)

    def training_step(self, batch: TensorDict, **kwargs):
        return self.step(batch)

    def validation_step(self, batch: TensorDict, **kwargs):
        return self.step(batch)

    def test_step(self, batch: TensorDict, **kwargs):
        return self.step(batch)

    def predict_step(self, batch: TensorDict):
        return self.step(batch)

    def step(self, batch: TensorDict):
        out = self.forward(batch)
        y_hat: torch.Tensor = out[self.out_key]

        if self.target_key in batch:  # Offline
            a = batch[self.target_key]
            loss = self.criterion(a, y_hat)
            return {"loss": loss, "pred": y_hat}
        else:  # Online
            if self.topk >= 0:
                v, _ = y_hat.topk(self.topk)
                y_hat[y_hat < v[:, [-1]]] = -math.inf

            probs = y_hat.softmax(dim=-1)

            # FIXME: This only allows one single action at a time
            if self.do_sample:
                idx = probs.multinomial(num_samples=1)
            else:
                idx = probs.argmax()

            a = torch.nn.functional.one_hot(idx, num_classes=probs.shape[-1])
            out[self.out_key][:, -1] = a  # Replace prediction by fixed value

            return {"pred": out}
    
    def on_epoch_start(
        self, dataloader: Any
    ):
        if not isinstance(dataloader, GymnasiumDataloader):
            return
        
        dataloader.play()
        