from abc import abstractmethod
from functools import partial
from typing import Callable, Final, Iterable, NamedTuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import IterableDataset
from torchrl.collectors import SyncDataCollector

from pretrained.models import ArnoldAgent
from src.data.dataset import DynamicGymnasiumDataset
from src.data.env import make_env

BATCH_SIZE = 8
BATCH_TRAJ_LEN = 64  # TODO 192 / 3
NUM_TRAJS = 2  # TODO


class DoomDataModule(LightningDataModule):
    class DatasetInfo(NamedTuple):
        name: str
        policy_maker: torch.nn.Module
        create_env_fn: Callable
        size: int
        max_steps_per_traj: int
        target_return_scaling_factor: float = 1.5

    def __init__(self, *, batch_size=8, batch_traj_len=64, num_workers=0):
        self.batch_size: Final = batch_size
        self.batch_traj_len: Final = batch_traj_len
        self.num_workers: Final = num_workers
        self.num_trajs: Final = NUM_TRAJS  # TODO

        self.max_seen_rewards: dict[str, np.float64] = {}

    def train_dataloader(self) -> Iterable[IterableDataset]:
        for dataset_info in self.get_datasets():
            max_seen_reward=self.max_seen_rewards.get(dataset_info.name)
            if max_seen_reward is not None:
                max_seen_reward *= dataset_info.target_return_scaling_factor
            
            dataset = DynamicGymnasiumDataset(
                size=dataset_info.size,
                batch_size=self.batch_size,
                batch_traj_len=self.batch_traj_len,
                max_traj_len=dataset_info.max_steps_per_traj,
                num_trajs=self.num_trajs,
                policy=dataset_info.policy_maker(),
                collector_maker=SyncDataCollector,  # TODO
                num_workers=self.num_workers,
                create_env_fn=dataset_info.create_env_fn,
                max_seen_reward=max_seen_reward,
            )

            yield dataset

            self.max_seen_rewards[dataset_info.name] = dataset.max_seen_reward

    @abstractmethod
    def get_datasets(self) -> list[DatasetInfo]: ...


class DoomOfflineDataModule(DoomDataModule):
    def __init__(
        self,
        *,
        batch_size=8,
        batch_traj_len=64,
        num_workers=0,
        max_seen_rewards: dict[str, np.float64] | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            batch_traj_len=batch_traj_len,
            num_workers=num_workers,
        )

        self.max_seen_rewards = max_seen_rewards or {}

    def get_datasets(self):
        return [
            super().DatasetInfo(
                name="defend_the_center",
                policy_maker=partial(ArnoldAgent, "defend_the_center"),
                create_env_fn=partial(make_env, "sa/arnold/DefendCenter-v0"),
                size=100,  # TODO
                max_steps_per_traj=1000,  # TODO
            )
        ]


class DoomOnlineDataModule(DoomDataModule):
    def __init__(
        self,
        policy: Callable,
        *,
        batch_size: int = 8,
        batch_traj_len: int = 64,
        num_workers: int = 0,
    ):
        self.policy = policy

        self.batch_size: Final = batch_size
        self.batch_traj_len: Final = batch_traj_len
        self.num_workers: Final = num_workers
        self.num_trajs: Final = NUM_TRAJS  # TODO

    def get_datasets(self):
        return [
            super().DatasetInfo(
                lambda: self.policy,
                partial(make_env, "sa/arnold/DefendCenter-v0"),
                size=100,  # TODO
                max_steps_per_traj=1000,  # TODO
            )
        ]
