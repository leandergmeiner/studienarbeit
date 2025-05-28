import inspect
import random
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable, Iterator, Literal

import dill  # noqa: F401
import numpy as np
import torch
from lightning import LightningDataModule
from tensordict.nn import TensorDictModule
from torch.utils.data import DataLoader, IterableDataset
from torchrl.envs import Compose, GymEnv, SerialEnv, TransformedEnv

from pretrained.models import ArnoldAgent
from src.data.streaming import GymnasiumStreamingDataset, LazyChainDataset
from src.data.transforms import (
    arnold_dataset_make_transforms,
    arnold_env_make_transforms,
    online_dataset_make_transforms,
    online_env_make_transforms,
)


@dataclass
class DatasetInfo:
    name: str
    create_env_fn: Callable
    max_steps: int
    max_steps_per_traj: int
    policy_maker: torch.nn.Module | None = None
    make_env_transforms: Callable[..., list] = list
    make_dataset_transforms: Callable[..., list] = list
    target_return_scaling_factor: float = 1.5
    method: Literal["offline", "online"] = "offline"


_DOOM_DATASETS = [
    DatasetInfo(
        name="defend_the_center",
        policy_maker=partial(ArnoldAgent, "defend_the_center"),
        create_env_fn=partial(GymEnv, "sa/ArnoldDefendCenter-v0"),
        make_env_transforms=arnold_env_make_transforms,
        make_dataset_transforms=partial(
            arnold_dataset_make_transforms,
            observation_shape=(224, 224),
            exclude_next_observation=True,
            collector_out_key="action",
            rtg_key="target_return",
        ),
        max_steps=1_000_000,
        max_steps_per_traj=500,
    ),
    DatasetInfo(
        name="defend_the_center",
        method="online",
        policy_maker=None,  # Online
        create_env_fn=partial(GymEnv, "sa/ArnoldDefendCenter-v0"),
        # TODO:
        make_env_transforms=online_env_make_transforms,
        make_dataset_transforms=partial(
            online_dataset_make_transforms,
            observation_shape=(224, 224),
            rtg_key="target_return",
        ),
        max_steps=250_000,
        max_steps_per_traj=500,
    ),
]
_NUM_ACTIONS = 10
# _FRAME_SKIP = 4  # TODO: This number is not enforced


class DoomStreamingDataModule(LightningDataModule):
    NUM_ACTIONS = _NUM_ACTIONS
    # FRAME_SKIP = _FRAME_SKIP

    def __init__(
        self,
        method: Literal["offline", "online"] = "offline",
        policy: TensorDictModule | None = None,
        batch_size: int | None = None,
        batch_traj_len: int = 64,
        num_workers: int = 0,
        num_trajs: int = 0,
        max_seen_rtgs: dict[str, np.float64] | None = None,
        pin_memory=torch.cuda.is_available(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["policy"])

        self._batch_size = batch_size
        self.batch_traj_len = batch_traj_len
        self.num_workers = num_workers
        self.num_trajs = num_trajs or batch_size
        self.policy = policy
        self.method: Literal["offline", "online"] = method
        self.pin_memory = pin_memory
        self.max_seen_rtgs = max_seen_rtgs or {}

        # Needed for state loading
        self._start_index = 0
        self._dataset_start_index = 0

    def set_mode(self, method: Literal["offline", "online"], policy: TensorDictModule):
        self.method = method
        self.policy = policy

    def setup(self, stage):
        # self.stage = stage
        datasets = deepcopy(_DOOM_DATASETS)
        datasets = filter(lambda d: d.method == self.method, datasets)
        for d in datasets:
            if isinstance(d.policy_maker, partial):
                func = d.policy_maker.func
            else:
                func = d.policy_maker

            if "batch_size" in inspect.getfullargspec(func).args:
                d.policy_maker = partial(d.policy_maker, batch_size=1)

        total_length = sum(
            dataset.max_steps // (self.batch_size * self.batch_traj_len)
            for dataset in datasets
        )
        self._dataset = LazyChainDataset(
            partial(self._dataset_iterator, datasets), total_length=total_length
        )

    def _dataset_iterator(
        self, datasets: Iterable[DatasetInfo]
    ) -> Iterator[IterableDataset]:
        random.shuffle(datasets)

        for dataset_info in datasets:
            max_seen_rtg = self.max_seen_rtgs.get(dataset_info.name)
            if max_seen_rtg is not None:
                max_seen_rtg *= dataset_info.target_return_scaling_factor

            def make_env_serial():
                def make_env():
                    env = dataset_info.create_env_fn()
                    wrapped = TransformedEnv(
                        env,
                        Compose(
                            *dataset_info.make_env_transforms(
                                target_return=max_seen_rtg
                            )
                        ),
                    )
                    return wrapped

                return SerialEnv(1, make_env)

            # _dataset_start_index is not 0 if we loaded from a state dict
            size = dataset_info.max_steps - self._dataset_start_index

            def online_policy():
                return self.policy

            if dataset_info.policy_maker:
                policy_maker = dataset_info.policy_maker()
                torch._dynamo.config.capture_scalar_outputs = True
                policy = policy_maker()
                policy = torch.compile(policy)
                torch._dynamo.config.capture_scalar_outputs = False
            else:
                policy = online_policy()

            policy = torch.no_grad(policy)

            dataset = GymnasiumStreamingDataset(
                size=size,
                batch_size=self.batch_size,
                batch_traj_len=self.batch_traj_len,
                max_traj_len=dataset_info.max_steps_per_traj,
                num_trajs=self.num_trajs,
                num_workers=self.num_workers,
                policy=policy,
                max_seen_rtg=max_seen_rtg,
                create_env_fn=make_env_serial,
                transform=dataset_info.make_dataset_transforms(),
                compilable=True,
            )

            self.current_dataset = dataset

            yield dataset

            self._start_index += 1

            self.max_seen_rtgs[dataset_info.name] = dataset.max_seen_rtg

    def _dataloader(self):
        return DataLoader(
            self._dataset,
            collate_fn=torch.cat,
            in_order=False,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            pin_memory=self.pin_memory,
            pin_memory_device="cuda:0" if self.pin_memory else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader()

    @property
    def batch_size(self):
        return self._batch_size | self.hparams.batch_size

    def state_dict(self):
        return {
            "index": self._start_index,
            "dataset_index": self._dataset_start_index,
            "max_seen_rtgs": self.max_seen_rtgs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._start_index = state_dict["index"]
        self._dataset_start_index = state_dict["dataset_index"]
        self.max_seen_rtgs = state_dict["max_seen_rtgs"]
