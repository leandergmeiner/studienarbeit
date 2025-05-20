from typing import Literal
import warnings

import lightning as L
import torch
from torch.nn import ModuleDict
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, dispatch
from typing import Optional

import torchrl
from src.modules.types import TrajectoryModel
from torchrl.modules import (
    ProbabilisticActor,
    DecisionTransformerInferenceWrapper
)
from tensordict.nn.probabilistic import InteractionType
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
from transformers import AutoModel, GPT2Config, GPT2Model

from einops import rearrange


class DTActor(torch.nn.Module):
    def __init__(self, model: TrajectoryModel, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = model
        self.action_layer = torch.nn.Linear(hidden_dim, action_dim)
        self.ln = torch.nn.LayerNorm(hidden_dim)

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
        hidden_state = self.ln(hidden_state)
        return self.action_layer(hidden_state)


class DTInferenceWrapper(DecisionTransformerInferenceWrapper):
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


class SpatialTransformerEncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        frame_skip: int = 4,
        resolution: tuple[int, int] | None = (224, 224),
    ):
        from src.modules.conv import PatchEmbedding
        from inspect import getfullargspec

        super().__init__()
        model = AutoModel.from_pretrained(
            model_name_or_path, attn_implementation="sdpa", add_pooling_layer=False
        )

        self.encoder = model.encoder

        if "resolution" in getfullargspec(self.encoder.forward).args:
            self.resolution = resolution
        else:
            self.resolution = None

        self.hidden_size = model.encoder.layer[0].layernorm_after.normalized_shape[0]
        base_conv = model.embeddings.patch_embeddings.projection

        self.patching = PatchEmbedding(
            base_conv,
            frame_skip,
            num_patches=model.embeddings.patch_embeddings.num_patches,
            method="uniform",
            additional_distillation_token=not self.resolution,
        )

    def forward(self, observations: torch.Tensor):
        encoder = torch.vmap(self.encoder, in_dims=-4, randomness="different")
        outputs = self.patching(observations)
        if self.resolution is not None:
            return encoder(outputs, resolution=self.resolution)
        else:
            return encoder(outputs)


class SpatialCNNEncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int = 192,
        encoder: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder or AutoModel.from_pretrained(model_name_or_path)
        self.linear = torch.nn.LazyLinear(output_dim)

    def forward(self, observations: torch.Tensor):
        observations = torch.stack(
            [
                self.encoder(obs)["last_hidden_state"]
                for obs in observations.unbind(dim=-4)
            ],
            dim=-4,
        )
        observations = rearrange(observations, "... d w h -> ... (d w h)")
        observations = self.linear(observations)
        return rearrange(observations, "... d -> ... 1 d")


class LightningSequenceActor(L.LightningModule):
    def __init__(
        self,
        num_actions: int,
        inference_context: int = 64,
        lr=0.001,
        action_key="action",
        out_action_key="action",
        observation_key="observation",
        rtg_key="return_to_go",
        labels_key="labels",
        model: torch.nn.Module | None = None,
        criterion: torch.nn.Module | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])

        self.action_key = action_key
        self.out_action_key = out_action_key
        self.observation_key = observation_key
        self.rtg_key = rtg_key
        self.labels_key = labels_key

        self.lr = lr

        self.model = model or self.default_model(
            num_actions=num_actions, inference_context=inference_context, **kwargs
        )

        self._actor = ProbabilisticActor(
            TensorDictModule(
                self.model,
                in_keys=[self.observation_key, self.action_key, self.rtg_key],
                out_keys=["logits"],
            ),
            in_keys=["logits"],
            out_keys=[self.out_action_key],
            distribution_class=torch.distributions.OneHotCategorical,
            # distribution_class=torchrl.modules.TanhDelta,
            # distribution_class=RelaxedBernoulli,
            # distribution_kwargs=dict(temperature=1.0), # TODO
            default_interaction_type=InteractionType.RANDOM,
        )

        self._inference_actor = DTInferenceWrapper(
            self._actor, inference_context=inference_context
        )

        self._inference_actor.set_tensor_keys(
            observation=self.observation_key,
            action=self.action_key,
            return_to_go=self.rtg_key,
            out_action=self.out_action_key,
        )

        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.metrics = ModuleDict(
            {
                "auroc": MultilabelAUROC(num_actions),
                "accuracy": MultilabelAccuracy(num_actions),
            }
        )

        self._used_actor = None

    @dispatch
    def forward(self, tensordict: TensorDict) -> TensorDict:
        # TODO: This is whack
        tensordict = tensordict.clone(False)
        if tensordict.batch_size:
            tensordict.batch_size = tensordict.batch_size[:-1]

        out: TensorDict = self._used_actor(tensordict)
        return out

    def training_step(self, batch: TensorDict):
        # FIXME: Use (collector, mask) for gradient computation
        labels = batch[self.labels_key]
        out = self.forward(batch)
        logits = out["logits"]

        if ("collector", "mask") in batch:
            mask = batch[("collector", "mask")]
            logits = logits[mask]
            labels = labels[mask]

        loss: torch.Tensor = self.criterion(logits, labels)
        metrics = self._calculate_metrics(logits, labels)

        self.log("loss", loss, prog_bar=True)
        self.log_dict(metrics)

        return {"loss": loss}

    def on_train_start(self):
        self._used_actor = self._actor

    def on_train_end(self):
        self._used_actor = self._inference_actor
        
    def on_predict_start(self):
        self._used_actor = self._inference_actor

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=(self.lr or self.learning_rate)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}

    def _calculate_metrics(self, prediction: torch.Tensor, label: torch.Tensor):
        label = label.int()
        return {key: metric(prediction, label) for key, metric in self.metrics.items()}

    def state_dict(self):
        state_dict = super().state_dict()
        return {k: v for k, v in state_dict.items() if "actor" not in k.split(".")[0]}

    @property
    def device(self):
        return self._actor.device

    @property
    def in_keys(self):
        return [self.observation_key, self.action_key, self.rtg_key]

    @property
    def out_keys(self):
        return sorted(
            set(self._actor.out_keys).union(
                {self.observation_key, self.action_key, self.rtg_key}
            ),
            key=str,
        )

    @classmethod
    def default_model(
        cls,
        method: Literal["transformer", "cnn"] = "transformer",
        frame_skip: int = 1,
        num_actions: int = 10,
        inference_context: int = 64,
        resolution: tuple[int, int] = (224, 224),
    ) -> DTActor:
        from src.modules.models import VideoDT

        # TODO: Image Processor

        hidden_size = 256

        if method == "transformer":
            spatial_encoder = SpatialTransformerEncoderWrapper(
                # "facebook/deit-small-distilled-patch16-224",
                "microsoft/beit-base-patch16-224",
                frame_skip,
                resolution,
            )
        elif method == "cnn":
            if frame_skip != 1:
                warnings.warn(
                    f"Frame skip is specified as {frame_skip} but {method=} does not use it."
                )
                frame_skip = 1

            # This size was chosen to have the model comparable to the transformer approach
            spatial_encoder = SpatialCNNEncoderWrapper(
                "microsoft/resnet-50", hidden_size
            )

        spatial_encoder = spatial_encoder.train()

        temporal_transformer = GPT2Model(
            GPT2Config(
                vocab_size=1,
                n_embd=hidden_size,
                n_positions=3 * inference_context,
                n_layer=12,
                n_head=8,
                attn_pdrop=0.1,
                embd_pdrop=0.1,
                resid_pdrop=0.1,
                use_flash_attention_2=True,
            )
        )

        transformer = DTActor(
            VideoDT(
                hidden_size=hidden_size,
                # patching=patching,
                frame_skip=frame_skip,
                spatial_encoder=spatial_encoder,
                temporal_transformer=temporal_transformer,
            ),
            hidden_dim=hidden_size,
            action_dim=num_actions,
        )

        return transformer
