from typing import Callable, Final, Iterable, NamedTuple
from abc import abstractmethod

import torch
from functools import partial
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from src.data.dataset import DynamicGymnasiumDataset
from src.data.env import make_env

from pretrained.models import ArnoldAgent

from torchrl.data.replay_buffers import LazyTensorStorage

from lightning import LightningDataModule
from torch.utils.data import IterableDataset


BATCH_SIZE = 8
BATCH_TRAJ_LEN = 64  # 192 / 3
NUM_TRAJS = 2  # TODO


class DoomDataModule(LightningDataModule):
    class DatasetInfo(NamedTuple):
        policy_maker: torch.nn.Module
        create_env_fn: Callable
        size: int
        max_steps_per_traj: int

    def __init__(self, *, batch_size=8, batch_traj_len=64, num_workers=0):
        self.batch_size: Final = batch_size
        self.batch_traj_len: Final = batch_traj_len
        self.num_workers: Final = num_workers
        self.num_trajs: Final = NUM_TRAJS  # TODO

    def train_dataloader(self) -> Iterable[IterableDataset]:
        for dataset_info in self.get_datasets():
            yield DynamicGymnasiumDataset(
                size=dataset_info.size,
                batch_size=self.batch_size,
                batch_traj_len=self.batch_traj_len,
                max_traj_len=dataset_info.max_steps_per_traj,
                num_trajs=self.num_trajs,
                policy=dataset_info.policy_maker(),
                collector_maker=SyncDataCollector,  # TODO
                num_workers=self.num_workers,
                create_env_fn=dataset_info.create_env_fn,
            )

    @abstractmethod
    def get_datasets(self) -> list[DatasetInfo]: ...


class DoomOfflineDataModule(DoomDataModule):
    def __init__(self, *, batch_size=8, batch_traj_len=64, num_workers=0):
        super().__init__(
            batch_size=batch_size,
            batch_traj_len=batch_traj_len,
            num_workers=num_workers,
        )

    def get_datasets(self):
        return [
            super().DatasetInfo(
                partial(ArnoldAgent, "defend_the_center"),
                partial(make_env, "sa/arnold/DefendCenter-v0"),
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


def get_online_datasets(online_policy: Callable):
    online_policy = torch.no_grad(online_policy)

    datasets: Final[list[Callable[..., DynamicGymnasiumDataset]]] = [
        partial(
            DynamicGymnasiumDataset,
            size=100,  # TODO
            batch_size=BATCH_SIZE,
            batch_traj_len=BATCH_TRAJ_LEN,
            max_traj_len=1000,
            num_trajs=10,  # TODO
            collector_maker=partial(SyncDataCollector, policy=online_policy),
            create_env_fn=partial(make_env, "sa/DefendLine-v0"),
        )
    ]

    for dataset_maker in datasets:
        yield dataset_maker()
