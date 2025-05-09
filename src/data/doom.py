from abc import abstractmethod
from functools import partial
from typing import Callable, Final, Iterable, NamedTuple, Any

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import IterableDataset, DataLoader
from torchrl.collectors import SyncDataCollector

from pretrained.models import ArnoldAgent
from src.data.env import make_env
from src.data.streaming import GymnasiumStreamingDataset, LazyChainDataset

BATCH_SIZE = 8
BATCH_TRAJ_LEN = 64  # TODO 192 / 3
NUM_TRAJS = 2  # TODO


class StreamingDataModule(LightningDataModule):
    class DatasetInfo(NamedTuple):
        name: str
        policy_maker: torch.nn.Module
        create_env_fn: Callable
        size: int
        max_steps_per_traj: int
        target_return_scaling_factor: float = 1.5

    def __init__(self, *, batch_size=8, batch_traj_len=64, num_workers=0):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size: Final = batch_size
        self.batch_traj_len: Final = batch_traj_len
        self.num_workers: Final = num_workers
        self.num_trajs: Final = NUM_TRAJS  # TODO

        self.max_seen_rtgs: dict[str, np.float64] = {}

        # Needed for state loading
        self._start_index = 0
        self._dataset_start_index = 0

    def _train_dataset_iterator(self) -> Iterable[IterableDataset]:
        for dataset_info in self.get_datasets()[self._start_index :]:
            max_seen_rtg = self.max_seen_rtgs.get(dataset_info.name)
            if max_seen_rtg is not None:
                max_seen_rtg *= dataset_info.target_return_scaling_factor

            create_env_fn = dataset_info.create_env_fn
            create_env_fn = partial(create_env_fn, num_workers=self.num_workers)

            # _dataset_start_index is not 0 if we loaded from a state dict
            size = dataset_info.size - self._dataset_start_index

            dataset = GymnasiumStreamingDataset(
                size=size,
                batch_size=self.batch_size,
                batch_traj_len=self.batch_traj_len,
                max_traj_len=dataset_info.max_steps_per_traj,
                num_trajs=self.num_trajs,  # TODO: Maybe self.batch_size or self.num_workers?
                policy=dataset_info.policy_maker(),
                collector_maker=SyncDataCollector,  # TODO
                num_workers=self.num_workers,
                create_env_fn=create_env_fn,
                max_seen_rtg=max_seen_rtg,
                make_transform_kwargs=dict(observation_shape=(224, 224)),
            )

            yield dataset

            self.max_seen_rtgs[dataset_info.name] = dataset.max_seen_rtg

    def _dataloader(self, datasets: Iterable[IterableDataset]):
        return DataLoader(LazyChainDataset(datasets), batch_size=None, collate_fn=lambda x: x)


    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self._train_dataset_iterator())
    
    @abstractmethod
    def get_datasets(self) -> list[DatasetInfo]: ...

    def state_dict(self):
        return {
            "index": self._start_index,
            "dataset_index": self._dataset_start_index,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._start_index = state_dict["index"]
        self._dataset_start_index = state_dict["dataset_index"]


class DoomOfflineDataModule(StreamingDataModule):
    def __init__(
        self,
        *,
        batch_size=BATCH_SIZE,
        batch_traj_len=BATCH_TRAJ_LEN,
        num_workers=0,
        max_seen_rewards: dict[str, np.float64] | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            batch_traj_len=batch_traj_len,
            num_workers=num_workers,
        )

        self.max_seen_rtgs = max_seen_rewards or {}

    def get_datasets(self):
        return [
            super().DatasetInfo(
                name="defend_the_center",
                policy_maker=partial(ArnoldAgent, "defend_the_center"),
                create_env_fn=partial(make_env, "sa/ArnoldDefendCenter-v0"),
                size=self.batch_size,  # TODO
                max_steps_per_traj=500,  # TODO
            )
        ]


class DoomOnlineDataModule(StreamingDataModule):
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
                partial(make_env, "sa/ArnoldDefendCenter-v0"),
                size=100,  # TODO
                max_steps_per_traj=1000,  # TODO
            )
        ]
