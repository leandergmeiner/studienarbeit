import warnings
from typing import Literal

import lightning as L
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, dispatch
from tensordict.nn.probabilistic import InteractionType
from torch.nn import ModuleDict
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
from transformers import GPT2Config, GPT2Model

# from torchao.float8 import Float8LinearConfig, convert_to_float8_training
# from torchrl.modules.tensordict_module import SafeSequential
from torchrl.modules import (
    SafeProbabilisticTensorDictSequential,
    SafeProbabilisticModule,
)
from torchrl.objectives import OnlineDTLoss

from src.modules.modules import (
    DTInferenceWrapper,
    OnlineDTActor,
    SpatialCNNEncoderWrapper,
    SpatialTransformerEncoderWrapper,
    VideoDT,
)
from tensordict.nn import set_interaction_type
from timm.optim.lamb import Lamb


class LightningSequenceActor(L.LightningModule):
    def __init__(
        self,
        num_actions: int,
        inference_context: int = 64,
        lr=0.001,
        model_type: Literal["transformer", "cnn"] = "transformer",
        frame_skip: int = 1,
        warmup_step: int = 30,
        init_temperature: float = 0.1,
        action_key="action",
        out_action_key="action",
        observation_key="observation",
        rtg_key="return_to_go",
        target_key="target_action",
        model: torch.nn.Module | None = None,
        method: Literal["offline", "online"] = "offline",
        accumulate_grad_batches: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "accumulate_grad_batches"])
        self.automatic_optimization = False

        self.action_key = action_key
        self.out_action_key = out_action_key
        self.observation_key = observation_key
        self.rtg_key = rtg_key
        self.target_key = target_key

        self.warmup_steps = warmup_step

        self.model_type = model_type
        self.frame_skip = frame_skip
        self.num_actions = num_actions
        self.inference_context = inference_context

        self.method: Literal["offline", "online"] = method
        self.init_temperature = init_temperature

        self.accumulate_grad_batches = accumulate_grad_batches
        self.lr = lr

        # Don't declare this as a Parameter, we perform gradient descent on it on its own
        # and don't use the normal optimizer

        self.log_temperature = None
        self.model = model
        self._training_actor = None
        self._inference_actor = None

        self.metrics = ModuleDict(
            {
                "auroc": MultilabelAUROC(num_actions),
                "accuracy": MultilabelAccuracy(num_actions),
            }
        )

    @set_interaction_type(InteractionType.RANDOM)
    @dispatch
    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self._reshape_batch(tensordict)
        out: TensorDict = self.inference_actor(tensordict)
        return out

    def _reshape_batch(self, tensordict: TensorDict):
        # TODO: This is whack
        tensordict = tensordict.clone(False)
        if tensordict.batch_size:
            tensordict.batch_size = tensordict.batch_size[:-1]
        return tensordict

    def training_step(self, batch: TensorDict, batch_idx: int):
        opt, temp_opt = self.optimizers()

        # UPDATE TEMPERATURE
        x: SafeProbabilisticModule = self.training_actor[-1]
        x.distribution_kwargs.update(temperature=self.temperature.detach())

        # FIXME: Use (collector, mask) for gradient computation
        batch = self._reshape_batch(batch)

        interaction_type = (
            InteractionType.RANDOM
            if self.method == "online"
            else InteractionType.DETERMINISTIC
        )

        with set_interaction_type(interaction_type):
            loss = self.loss_module(batch)

        # GENERAL OPTIMIZER
        loss_value = loss["loss_log_likelihood"]
        loss_value /= self.accumulate_grad_batches

        opt.zero_grad()
        self.manual_backward(loss_value)

        # TEMPERATURE OPTIMIZER
        entropy_delta: torch.Tensor = (
            loss["entropy"] - self.loss_module.target_entropy
        )
        temperature_loss = self.temperature * entropy_delta.detach()
        temperature_loss /= self.accumulate_grad_batches

        temp_opt.zero_grad()
        self.manual_backward(temperature_loss)

        # OPTIMIZER STEP
        if batch_idx + 1 % self.accumulate_grad_batches == 0:
            opt.step()
            temp_opt.step()

        # COMPUTE METRICS
        logits = loss[self.loss_module.tensor_keys.action_pred]
        labels = loss[self.loss_module.tensor_keys.action_target]
        metrics = self._calculate_metrics(logits, labels)

        self.log("loss", loss_value, prog_bar=True)
        self.log_dict(metrics)

    @set_interaction_type(InteractionType.RANDOM)
    def predict_step(self, batch: TensorDict):
        return self(batch)

    def setup_method(self, method: Literal["offline", "online"]):
        self.method = method

    def _optimizer_step_lambda(self, step: int):
        return min(step / self.warmup_steps, 1.0)

    def configure_optimizers(self):
        lr = self.lr or self.learning_rate

        params = [
            param for param in self.parameters() if param is not self.log_temperature
        ]

        optimizer = Lamb(params, lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self._optimizer_step_lambda
        )

        optim_config = {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

        log_temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=lr)

        log_temperature_optim_config = {"optimizer": log_temperature_optimizer}

        return optim_config, log_temperature_optim_config

    def configure_model(self):
        if self.model is not None:
            return

        model = self._default_model(
            self.model_type,
            self.frame_skip,
            self.num_actions,
            self.inference_context,
        )

        # Sadly supported on the GPUs available to us
        # float8_config = Float8LinearConfig(
        #     pad_inner_dim=True,
        # )
        # convert_to_float8_training(model, config=float8_config)
        # model = torch.compile(model)

        self.model = model
        self.log_temperature = torch.tensor(self.init_temperature).log()
        self.log_temperature.requires_grad = True

        self._configure_actors()

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
        self.target_key = labels or self.target_key
        self._configure_actors()

    def state_dict(self):
        state_dict = super().state_dict()
        return {k: v for k, v in state_dict.items() if "actor" not in k.split(".")[0]}

    def _configure_actors(self):
        actor = SafeProbabilisticTensorDictSequential(
            self.model,
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["logits"],
                distribution_class=torch.distributions.Normal,
                return_log_prob=True,
            ),
            SafeProbabilisticModule(
                in_keys=["logits"],
                out_keys=[self.out_action_key],
                distribution_class=torch.distributions.RelaxedOneHotCategorical,
                distribution_kwargs=dict(temperature=self.temperature.detach()),
            ),
        )

        self._training_actor = actor
        self._inference_actor = DTInferenceWrapper(
            actor, inference_context=self.inference_context
        )
        self._inference_actor.set_tensor_keys(
            observation=self.observation_key,
            action=self.action_key,
            return_to_go=self.rtg_key,
            out_action=self.out_action_key,
        )

        target_entropy = -self.num_actions
        loss_module = OnlineDTLoss(self.training_actor, target_entropy=target_entropy)
        loss_module.tensor_keys.action_pred = self.out_action_key
        loss_module.tensor_keys.action_target = self.target_key
        self.loss_module = loss_module

    def _calculate_metrics(self, prediction: torch.Tensor, label: torch.Tensor):
        label = label.int()
        return {key: metric(prediction, label) for key, metric in self.metrics.items()}

    @property
    def device(self):
        return self._training_actor.device

    @property
    def in_keys(self):
        return [self.observation_key, self.action_key, self.rtg_key]

    @property
    def out_keys(self):
        return sorted(
            set(self._training_actor.out_keys).union(
                {self.observation_key, self.action_key, self.rtg_key}
            ),
            key=str,
        )

    @property
    def training_actor(self):
        return self._training_actor

    @property
    def inference_actor(self):
        return self._inference_actor

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def _default_model(
        self,
        model_type: Literal["transformer", "cnn"] = "transformer",
        frame_skip: int = 1,
        num_actions: int = 10,
        inference_context: int = 64,
        resolution: tuple[int, int] = (224, 224),
    ) -> TensorDictModule:
        # TODO: Image Processor

        hidden_size = 256

        if model_type == "transformer":
            spatial_encoder = SpatialTransformerEncoderWrapper(
                # "facebook/deit-small-distilled-patch16-224",
                "microsoft/beit-base-patch16-224",
                frame_skip,
                resolution,
            )
        elif model_type == "cnn":
            if frame_skip != 1:
                warnings.warn(
                    f"Frame skip is specified as {frame_skip} but {model_type=} does not use it."
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

        return TensorDictModule(
            transformer,
            in_keys=[self.observation_key, self.action_key, self.rtg_key],
            out_keys=["loc", "scale"],
        )
