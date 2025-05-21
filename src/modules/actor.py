import warnings
from typing import Literal

import lightning as L
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, dispatch
from tensordict.nn.probabilistic import InteractionType
from torch.nn import ModuleDict
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
from torchrl.modules import ProbabilisticActor
from transformers import GPT2Config, GPT2Model
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

from src.modules.modules import (
    DTInferenceWrapper,
    OnlineDTActor,
    SpatialCNNEncoderWrapper,
    SpatialTransformerEncoderWrapper,
    VideoDT,
)


class LightningSequenceActor(L.LightningModule):
    def __init__(
        self,
        num_actions: int,
        inference_context: int = 64,
        lr=0.001,
        method: Literal["transformer", "cnn"] = "transformer",
        frame_skip: int = 1,
        action_key="action",
        out_action_key="action",
        observation_key="observation",
        rtg_key="return_to_go",
        labels_key="labels",
        model: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])

        self.action_key = action_key
        self.out_action_key = out_action_key
        self.observation_key = observation_key
        self.rtg_key = rtg_key
        self.labels_key = labels_key

        self.method = method
        self.frame_skip = frame_skip
        self.num_actions = num_actions
        self.inference_context = inference_context

        self.lr = lr

        self.model = model
        self._actor = None
        self._inference_actor = None

        self.criterion = torch.nn.CrossEntropyLoss()
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

    def configure_model(self):
        if self.model is not None:
            return 

        model = self.default_model(
            self.method,
            self.frame_skip,
            self.num_actions,
            self.inference_context,
        )
        
        float8_config = Float8LinearConfig(
            pad_inner_dim=True,
        )

        convert_to_float8_training(model, config=float8_config)
        model = torch.compile(model)
        
        self.model = model
        self._configure_actors()

    def _configure_actors(self):
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
            self._actor, inference_context=self.inference_context
        )

        self._inference_actor.set_tensor_keys(
            observation=self.observation_key,
            action=self.action_key,
            return_to_go=self.rtg_key,
            out_action=self.out_action_key,
        )

    def set_tensor_keys(
        self,
        observation: str | None = None,
        action: str | None = None,
        return_to_go: str | None = None,
        out_action: str | None = None,
        labels: str | None = None,
    ):
        self.observation_key = observation or self.observation_key
        self.action_key = action or self.action_key
        self.rtg_key = return_to_go or self.rtg_key
        self.out_action_key = out_action or self.out_action_key
        self.labels_key = labels or self.labels_key
        self._configure_actors()

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
    ) -> OnlineDTActor:
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

        transformer = OnlineDTActor(
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
