import warnings
from typing import Literal

import lightning as L
import torch
from tensordict import TensorDict, unravel_key
from tensordict.nn import (
    TensorDictModule,
    TensorDictModuleBase,
    dispatch,
    set_interaction_type,
)
from tensordict.nn.probabilistic import InteractionType
from timm.optim.lamb import Lamb
from torch.nn import ModuleDict
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
from torchrl.envs import CatFrames

# from torchao.float8 import Float8LinearConfig, convert_to_float8_training
# from torchrl.modules.tensordict_module import SafeSequential
from torchrl.modules import (
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
    TanhNormal,
)
from torchrl.objectives import OnlineDTLoss
from transformers import GPT2Config, GPT2Model

from src.modules.modules import (
    DTInferenceWrapper,
    OnlineDTActor,
    SpatialCNNEncoderWrapper,
    SpatialTransformerEncoderWrapper,
    VideoDT,
)


# NOTE: Von Yannick: Hahaha, der Name ist lustig weil ... STEP-Wrapper hahaha!
class DecisionTransformerInferenceStepWrapper(TensorDictModuleBase):
    def __init__(
        self,
        actor: TensorDictModuleBase,
        n_steps: int | None = None,
        *,
        action_dims: list[int],
        action_keys: list | None = None,
        observation_key: str | None = None,
    ):
        self.observation_key = observation_key or "observation"
        self.action_dims = action_dims
        self.action_keys = action_keys
        self.n_steps = n_steps

        super().__init__()
        self.actor = actor
        self.reset()

    # Called by step
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Method used for rollout, this means it should only take a single step tensordict,
        since it takes care of keeping the last n steps in memory.
        """  # FIXME <- This is soooo whack, what if multiple environments?????
        orig_td = tensordict.clone()
        orig_batch_size = orig_td.batch_size
        orig_td.batch_size = ()

        # No batch, no time
        if tensordict.batch_dims == 0:
            tensordict = tensordict[None, None, ...]
        elif tensordict.batch_dims == 1:  # No time
            tensordict = tensordict[:, None, ...]

        assert tensordict.batch_dims >= 1
        tensordict.batch_size = tensordict.batch_size[:1]

        td = TensorDict(
            {
                action_key: torch.zeros((*tensordict.batch_size, 1, dim))
                for action_key, dim in zip(self.action_keys, self.action_dims)
            },
            batch_size=[],
        )

        tensordict = self.cat_frames_obs._call(tensordict)
        tensordict = self.cat_frames_other._call(tensordict)

        tensordict = td.update(tensordict)
        tensordict = self.cat_frames_action._call(tensordict)

        # tensordict should now have the form [b; t; ...]

        # CatFrames stacks the frames from the right to the left
        # while we need them from the left to the right
        for key in self.actor.in_keys:
            tensordict[key] = tensordict[key].flip((1,))

        # Needed because some transforms don't do the casting
        tensordict = tensordict.to(self.device)  # TODO: Whack!
        output = self.actor(tensordict)
        output = output.to(orig_td.device)

        for key, dim in zip(self.action_keys, self.action_dims):
            orig_td[key] = output[key].reshape(*orig_batch_size, dim)

        orig_td.batch_size = orig_batch_size
        return orig_td

    def reset(self):
        # TODO: Whack WTF??
        self.cat_frames_obs = CatFrames(
            self.n_steps, in_keys=[self.observation_key], dim=-4
        )
        self.cat_frames_other = CatFrames(
            self.n_steps,
            in_keys=[
                key
                for key in self.actor.in_keys
                if key != self.observation_key and key not in self.action_keys
            ],
            dim=-2,
        )

        # We save one action less than total frames,
        # since we want to predict the action based on the
        # current frame.
        # To achieve this we always add a zero action at the start
        self.cat_frames_action = CatFrames(
            self.n_steps, in_keys=self.action_keys, dim=-2
        )

    @property
    def in_keys(self):
        return self.actor.in_keys + [self.init_key]

    @in_keys.setter
    def in_keys(self, keys: list):
        self.actor.in_keys = keys

    @property
    def out_keys(self):
        return self.actor.out_keys + [self.counter_key]

    @out_keys.setter
    def out_keys(self, keys: list):
        self.actor.out_keys = keys

    @property
    def action_keys(self):
        action_keys = self.__dict__.get("_action_keys", None)
        if action_keys is None:

            def ends_with_action(key):
                if isinstance(key, str):
                    return key == "action"
                return key[-1] == "action"

            action_keys = [key for key in self.actor.out_keys if ends_with_action(key)]

            self.__dict__["_action_keys"] = action_keys
        return action_keys

    @action_keys.setter
    def action_keys(self, value):
        if value is None:
            return
        self.__dict__["_actor_keys_map_values"] = None
        if not isinstance(value, list):
            value = [value]
        self._action_keys = [unravel_key(key) for key in value]

    @property
    def device(self):
        return next(self.actor.parameters()).device


class LightningDecisionTransformer(L.LightningModule, TensorDictModuleBase):
    def __init__(
        self,
        num_actions: int,
        inference_context: int = 64,
        lr=0.001,
        model_type: Literal["transformer", "cnn"] = "transformer",
        frame_skip: int = 1,
        warmup_step: int = 30,
        init_temperature: float = 1.0,
        action_key="action",
        out_action_key="action",
        observation_key="observation",
        rtg_key="return_to_go",
        target_key="target_action",
        model: torch.nn.Module | None = None,
        method: Literal["offline", "online"] = "offline",
        accumulate_grad_batches: int = 1,
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

        self.method = method
        self.init_temperature = torch.tensor(init_temperature)

        self.accumulate_grad_batches = accumulate_grad_batches
        self.lr = lr

        # Don't declare this as a Parameter, we perform gradient descent on it on its own
        # and don't use the normal optimizer

        self.actor = model
        self._training_actor = None
        self._inference_actor = None

        self.metrics = ModuleDict(
            {
                "auroc": MultilabelAUROC(num_actions),
                "accuracy": MultilabelAccuracy(num_actions),
            }
        )

        # This is a single step example for the forward method
        self.example_input_array = TensorDict(
            {
                self.observation_key: torch.rand((3, 224, 224)),
                self.action_key: torch.rand((self.num_actions)),
                self.rtg_key: torch.rand((1)),
                self.target_key: torch.rand((self.num_actions)),
            }
        )

    @set_interaction_type(InteractionType.RANDOM)
    @dispatch
    def forward(self, tensordict: TensorDict) -> TensorDict:
        return self.inference_actor(tensordict)

    @dispatch
    def predict_step(self, tensordict: TensorDict):
        return self.forward(tensordict)

    def on_predict_epoch_start(self):
        self.inference_actor.reset()

    def on_predict_epoch_end(self):
        self.inference_actor.reset()

    def training_step(self, batch: TensorDict, batch_idx: int):
        opt, temp_opt = self.optimizers()
        batch = self._training_reshape_batch(batch)

        # FIXME: Use (collector, mask) for gradient computation

        # Update temperature for distribution
        # self.actor[-1].distribution_kwargs.update(
        #     temperature=torch.tensor(float(self.temperature))
        # )

        # with set_interaction_type(self._training_interaction_type):
        loss = self.loss_module(batch)

        accumulated_grad_batches = batch_idx % self.accumulate_grad_batches == 0

        def closure_loss():
            if accumulated_grad_batches:
                opt.zero_grad()
            loss_value: torch.Tensor = (
                loss["loss_log_likelihood"]
                + loss["loss_entropy"]
            ) / self.accumulate_grad_batches
            self.manual_backward(loss_value)
            self.clip_gradients(
                opt, gradient_clip_val=0.25, gradient_clip_algorithm="norm"
            )

        with opt.toggle_model(sync_grad=accumulated_grad_batches):
            opt.step(closure=closure_loss)

        def closure_loss_temperature():
            if accumulated_grad_batches:
                temp_opt.zero_grad()
            temperature_loss: torch.Tensor = (
                loss["loss_alpha"] / self.accumulate_grad_batches
            )
            self.manual_backward(temperature_loss)

        with temp_opt.toggle_model(sync_grad=accumulated_grad_batches):
            temp_opt.step(closure=closure_loss_temperature)

        # logits = loss[self.loss_module.tensor_keys.action_pred]
        # labels = loss[self.loss_module.tensor_keys.action_target]
        # metrics = self._calculate_metrics(logits, labels)

        self.log(
            "loss",
            (loss["loss_log_likelihood"] + loss["loss_entropy"])
            / self.accumulate_grad_batches,
            prog_bar=True,
        )
        self.log_dict(loss)
        # self.log_dict(metrics)

    def _training_reshape_batch(self, tensordict: TensorDict):
        if tensordict.batch_dims == 1:
            tensordict = tensordict[None, ...]
        # Currently the model expects a batch and time dimension
        assert tensordict.batch_dims == 2
        tensordict.batch_size = tensordict.batch_size[:-1]
        return tensordict

    def validation_step(self, batch: TensorDict):
        # Batch is taken from an online rollout
        # So here we just need to calculate the metrics.
        if batch.batch_dims == 0:
            batch = batch[None, None, ...]
        elif batch.batch_dims == 1:
            batch = batch[None, ...]

        assert batch.batch_dims == 2

        reward = batch[("next", "reward")]
        max_rewards = torch.stack(tuple(torch.max(t) for t in reward.unbind(0)))
        mean_max_reward = torch.mean(max_rewards)
        self.log("reward", mean_max_reward, on_step=True)

    def configure_optimizers(self):
        lr = self.lr or self.learning_rate

        optimizer = Lamb(
            self.parameters(), lr=lr, weight_decay=5e-4, grad_averaging=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=3000, eta_min=lr / 100
        )

        optim_config = {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

        log_temperature_optimizer = torch.optim.Adam(
            [self.loss_module.log_alpha], lr=lr
        )

        log_temperature_optim_config = {"optimizer": log_temperature_optimizer}

        return optim_config, log_temperature_optim_config

    def configure_model(self):
        if self.actor is not None:
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

        self.actor = SafeProbabilisticTensorDictSequential(
            model,
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["logits"],
                distribution_class=TanhNormal,
                distribution_kwargs=dict(low=0., high=1.)
            ),
            SafeProbabilisticModule(
                in_keys=["logits"],
                out_keys=[self.out_action_key],
                distribution_class=torch.distributions.OneHotCategoricalStraightThrough,
            ),
        )
        self._configure_actor_wrappers()

        # Populate the model
        self.to("cpu")
        _ = torch.no_grad(self.inference_actor.forward)(self.example_input_array)
        self.inference_actor.reset()

    def _configure_actor_wrappers(self):
        self._training_actor = self.actor
        inference_actor = DTInferenceWrapper(
            self.actor, inference_context=self.inference_context
        )
        inference_actor.set_tensor_keys(
            observation=self.observation_key,
            action=self.action_key,
            return_to_go=self.rtg_key,
            out_action=self.out_action_key,
        )
        self._inference_actor = DecisionTransformerInferenceStepWrapper(
            inference_actor,
            self.inference_context,
            action_keys=[self.action_key],
            action_dims=[self.num_actions],
            observation_key=self.observation_key,
        )

        target_entropy = -self.num_actions
        loss_module = OnlineDTLoss(
            self.training_actor,
            target_entropy=target_entropy,
            alpha_init=self.init_temperature,
            max_alpha=1.5 * self.init_temperature,
        )
        loss_module.set_keys(
            action_pred=self.out_action_key, action_target=self.target_key
        )
        self.loss_module = loss_module

    def set_tensor_keys(
        self,
        observation: str | None = None,
        action: str | None = None,
        return_to_go: str | None = None,
        out_action: str | None = None,
        target: str | None = None,
    ):
        self.observation_key = observation or self.observation_key
        self.action_key = action or self.action_key
        self.rtg_key = return_to_go or self.rtg_key
        self.out_action_key = out_action or self.out_action_key
        self.target_key = target or self.target_key

        in_keys = [self.observation_key, self.action_key, self.rtg_key]
        out_keys = [self.out_action_key]

        # TODO: Bit whacky here
        self.actor[0].in_keys = in_keys
        self.actor[-1].out_keys = out_keys
        self.inference_actor.in_keys = in_keys
        self.inference_actor.out_keys = out_keys
        self.loss_module.set_keys(
            action_pred=self.out_action_key, action_target=self.target_key
        )

    @property
    def out_keys(self):
        return [self.out_action_key]

    @property
    def in_keys(self):
        return [
            self.observation_key,
            self.action_key,
            self.rtg_key,
            self.target_key,
        ]

    def state_dict(self):
        state_dict = super().state_dict()

        # Avoid saving the model multiple times, since its included in each actor
        return {k: v for k, v in state_dict.items() if "actor" not in k.split(".")[0]}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        super().load_state_dict(state_dict, False, assign)
        self._configure_actor_wrappers()

    def _calculate_metrics(self, prediction: torch.Tensor, label: torch.Tensor):
        label = label.int()
        return {key: metric(prediction, label) for key, metric in self.metrics.items()}

    @property
    def device(self):
        return self.actor.device

    @property
    def training_actor(self):
        return self._training_actor

    @property
    def inference_actor(self):
        return self._inference_actor

    @property
    def temperature(self):
        if hasattr(self, "loss_module"):
            return self.loss_module.alpha
        else:
            return self.init_temperature

    @property
    def method(self) -> Literal["offline", "online"]:
        return self._method

    @method.setter
    def method(self, value: Literal["offline", "online"]):
        assert value in ["offline", "online"]
        self._method = value

    @property
    def _training_interaction_type(self):
        interaction_type_mapping = {
            "online": InteractionType.RANDOM,
            "offline": InteractionType.DETERMINISTIC,
        }
        return interaction_type_mapping[self.method]

    def reset(self):
        self.inference_actor.reset()

    def _default_model(
        self,
        model_type: Literal["transformer", "cnn"] = "transformer",
        frame_skip: int = 1,
        num_actions: int = 10,
        inference_context: int = 64,
        hidden_size: int = 192,
        resolution: tuple[int, int] = (224, 224),
    ) -> TensorDictModule:
        # TODO: Image Processor

        if model_type == "transformer":
            spatial_encoder = SpatialTransformerEncoderWrapper(
                "facebook/deit-small-distilled-patch16-224",
                # "microsoft/beit-base-patch16-224",
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

        # spatial_encoder = spatial_encoder.eval()
        # if model_type == "transformer":
        #     spatial_encoder.patching.train()

        # If frame_skip > 1 only every frame_skip'th action and reward is passed to the
        # transformer, since the observations are shrunken by a factor of frame_skip
        # To keep actual context size equal, we perform some scaling.
        temp_inference_context = inference_context // frame_skip

        temporal_transformer = GPT2Model(
            GPT2Config(
                vocab_size=1,
                n_embd=hidden_size,
                n_positions=3 * temp_inference_context,  # [(R, s, a)]
                n_layer=12,
                n_head=8,
                attn_pdrop=0.1,
                embd_pdrop=0.1,
                resid_pdrop=0.1,
                use_flash_attention_2=True,
            )
        )

        transformer_actor = OnlineDTActor(
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
            transformer_actor,
            in_keys=[self.observation_key, self.action_key, self.rtg_key],
            out_keys=["loc", "scale"],
        )
