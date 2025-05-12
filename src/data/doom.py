import math
from abc import abstractmethod, ABC
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchrl.collectors import SyncDataCollector

from pretrained.models import ArnoldAgent
from src.data.env import make_env
from src.data.streaming import GymnasiumStreamingDataset, LazyChainDataset
import random


@dataclass
class DatasetInfo:
    name: str
    create_env_fn: Callable
    size: int
    max_steps_per_traj: int
    policy_maker: torch.nn.Module | None = None
    target_return_scaling_factor: float = 1.5


# TODO: Increase max_steps_per_traj

DOOM_DATASETS = [
    DatasetInfo(
        name="defend_the_center",
        policy_maker=partial(ArnoldAgent, "defend_the_center"),
        create_env_fn=partial(make_env, "sa/ArnoldDefendCenter-v0"),
        size=1_000_000,
        max_steps_per_traj=500,  # TODO
    ),
    DatasetInfo(
        name="health_gathering",
        policy_maker=partial(ArnoldAgent, "health_gathering"),
        create_env_fn=partial(make_env, "sa/ArnoldHealthGathering-v0"),
        size=1_000_000,
        max_steps_per_traj=500,  # TODO
    ),
    # FIXME
    # DatasetInfo(
    #     name="shotgun",
    #     policy_maker=partial(ArnoldAgent, "shotgun"),
    #     create_env_fn=partial(make_env, "sa/ArnoldShotgun-v0"),
    #     size=1_000_000,
    #     max_steps_per_traj=750,  # TODO
    # ),
]


class StreamingDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        *,
        batch_size=4,
        batch_traj_len=64,
        num_workers=0,
        max_seen_rtgs: dict[str, np.float64] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.batch_traj_len = batch_traj_len
        self.num_workers = num_workers
        self.num_trajs = batch_size

        self.max_seen_rtgs = max_seen_rtgs or {}

        # Needed for state loading
        self._start_index = 0
        self._dataset_start_index = 0

    def _train_dataset_iterator(self) -> Iterator[IterableDataset]:
        datasets = list(self.datasets)
        random.shuffle(datasets)
        
        # In the second run
        # print(datasets)

        for dataset_info in datasets:
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
                num_trajs=self.num_trajs,
                policy=dataset_info.policy_maker(),
                collector_maker=partial(
                    SyncDataCollector,
                ),  # TODO
                num_workers=self.num_workers,
                create_env_fn=create_env_fn,
                max_seen_rtg=max_seen_rtg,
                make_transform_kwargs=dict(
                    observation_shape=(224, 224),
                    exclude_next_observation=True,
                ),  # TODO: This is whack
                compilable=True,
            )

            self.current_dataset = dataset

            yield dataset

            self._start_index += 1

            self.max_seen_rtgs[dataset_info.name] = dataset.max_seen_rtg
            
    def _dataloader(self, make_datasets: Iterator[IterableDataset]):
        self._start_index = 0
        return LazyChainDataset(make_datasets)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self._train_dataset_iterator)

    @property
    @abstractmethod
    def _datasets(self) -> list[DatasetInfo]: ...

    @property
    def datasets(self) -> list[DatasetInfo]:
        return self._datasets[self._start_index :]

    def state_dict(self):
        return {
            "index": self._start_index,
            "dataset_index": self._dataset_start_index,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._start_index = state_dict["index"]
        self._dataset_start_index = state_dict["dataset_index"]


class DoomOfflineDataModule(StreamingDataModule):
    def __init__(self, rounds=100, **kwargs):
        super().__init__(**kwargs)
        self.rounds = rounds

    @property
    def _datasets(self):
        datasets = deepcopy(DOOM_DATASETS)

        for d in datasets:
            d.size = math.ceil(d.size / self.rounds)

            # TODO: This is whack and only specific to Arnold Models
            d.policy_maker = partial(d.policy_maker, batch_size=self.num_workers)

        datasets = self.rounds * datasets

        return datasets


class DoomOnlineDataModule(StreamingDataModule):
    def __init__(
        self,
        policy: Callable,
        rounds=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy = policy
        self.rounds = rounds

    @property
    def _datasets(self):
        datasets = deepcopy(DOOM_DATASETS)

        for d in datasets:
            d.size = math.ceil(d.size / self.rounds)
            d.policy_maker = lambda: self.policy

        datasets = self.rounds * list(datasets)

        return datasets
