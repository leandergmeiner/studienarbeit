import lightning as L
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, dispatch
from typing import Optional

import torchrl
from src.modules.types import TrajectoryModel
from torchrl.modules import (
    ProbabilisticActor,
)
from tensordict.nn.probabilistic import InteractionType

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


class DTInferenceWrapper(torchrl.modules.DecisionTransformerInferenceWrapper):
    def __init__(
        self,
        policy: TensorDictModule,
        *,
        inference_context: int = 5,
        spec: Optional[torchrl.data.TensorSpec] = None,
        device: torch.device | None = None,
    ):
        super().__init__(
            policy, inference_context=inference_context, spec=spec, device=device
        )

    @staticmethod
    def _check_tensor_dims(reward, obs, action):
        if not (reward.shape[:-1] == obs.shape[:-3] == action.shape[:-1]) or not (
            obs.shape[-3] == 3
        ):
            raise ValueError("Mismatched tensor dimensions.")

    def mask_context(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Mask the context of the input sequences."""
        observation = tensordict.get(self.observation_key).clone()
        action = tensordict.get(self.action_key).clone()
        return_to_go = tensordict.get(self.return_to_go_key).clone()
        self._check_tensor_dims(return_to_go, observation, action)

        observation[..., : -self.inference_context, :, :, :] = 0
        action[..., : -(self.inference_context - 1), :] = (
            0  # as we add zeros to the end of the action
        )
        action = torch.cat(
            [
                action[..., 1:, :],
                torch.zeros(
                    *action.shape[:-2], 1, action.shape[-1], device=action.device
                ),
            ],
            dim=-2,
        )
        return_to_go[..., : -self.inference_context, :] = 0

        tensordict.set(self.observation_key, observation)
        tensordict.set(self.action_key, action)
        tensordict.set(self.return_to_go_key, return_to_go)
        return tensordict


class LightningSequenceActor(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        inference_context: int = 512,
        lr=0.001,
        metrics: dict[str, torch.nn.Module] | None = None,
        action_key="action",
        out_action_key="action2",
        observation_key="observation",
        rtg_key="return_to_go",
        labels_key="labels",
    ):
        super().__init__()

        self.action_key = action_key
        self.out_action_key = out_action_key
        self.observation_key = observation_key
        self.rtg_key = rtg_key
        self.labels_key = labels_key

        self.lr = lr

        self.transformer = model

        model = TensorDictModule(
            model,
            in_keys=[self.observation_key, self.action_key, self.rtg_key],
            out_keys=["logits"],
        )
        
        self.actor = ProbabilisticActor(
            model,
            in_keys=["logits"],
            out_keys=[self.out_action_key],
            distribution_class=torch.distributions.OneHotCategorical,
            # distribution_class=RelaxedBernoulli,
            # distribution_kwargs=dict(temperature=1.0), # TODO
            default_interaction_type=InteractionType.RANDOM,
        )

        self.inference_actor = DTInferenceWrapper(self.actor, inference_context=inference_context)

        self.inference_actor.set_tensor_keys(
            observation=self.observation_key,
            action=self.action_key,
            return_to_go=self.rtg_key,
            out_action=self.out_action_key,
        )

        self.criterion = criterion
        self.metrics = metrics or {}
        self.used_actor = None

    @dispatch
    def forward(self, tensordict: TensorDict) -> TensorDict:
        # TODO: This is whack
        tensordict = tensordict.clone(False)
        if tensordict.batch_size:
            tensordict.batch_size = tensordict.batch_size[:-1]

        out: TensorDict = self.used_actor(tensordict)
        return out

    def training_step(self, batch: TensorDict):
        labels = batch[self.labels_key]
        out = self.forward(batch)
        loss = self.criterion(out["logits"], labels)
        return {"loss": loss}

    def on_predict_start(self):
        self.used_actor = self.inference_actor
        
    def on_train_start(self):
        self.used_actor = self.actor
        
    def on_train_end(self):
        self.used_actor = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @property
    def device(self):
        return self.actor.device

    @property
    def in_keys(self):
        return [self.observation_key, self.action_key, self.rtg_key]

    @property
    def out_keys(self):
        return sorted(
            set(self.actor.out_keys).union(
                {self.observation_key, self.action_key, self.rtg_key}
            ),
            key=str,
        )
