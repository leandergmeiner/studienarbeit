from copy import deepcopy
from functools import partial
from typing import Any, Callable, Iterator, Literal, Iterable
from dataclasses import dataclass

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from pretrained.models import ArnoldAgent
from src.data.env import make_env
from src.data.streaming import GymnasiumStreamingDataset, LazyChainDataset
import random


@dataclass
class DatasetInfo:
    name: str
    create_env_fn: Callable
    max_steps: int
    max_steps_per_traj: int
    policy_maker: torch.nn.Module | None = None
    target_return_scaling_factor: float = 1.5


_DOOM_DATASETS = [
    DatasetInfo(
        name="defend_the_center",
        policy_maker=partial(ArnoldAgent, "defend_the_center", random_action_chance=0.1),
        create_env_fn=partial(make_env, "sa/ArnoldDefendCenter-v0"),
        max_steps=1_000_000,
        max_steps_per_traj=500,
    ),
    # DatasetInfo(
    #     name="health_gathering",
    #     policy_maker=partial(ArnoldAgent, "health_gathering"),
    #     create_env_fn=partial(make_env, "sa/ArnoldHealthGathering-v0"),
    #     size=1_000_000,
    #     max_steps_per_traj=500,  # TODO
    # ),
    # FIXME
    # DatasetInfo(
    #     name="shotgun",
    #     policy_maker=partial(ArnoldAgent, "shotgun"),
    #     create_env_fn=partial(make_env, "sa/ArnoldShotgun-v0"),
    #     size=1_000_000,
    #     max_steps_per_traj=750,  # TODO
    # ),
]
_NUM_ACTIONS = 10
_FRAME_SKIP = 4  # TODO: This number is not enforced


class DoomStreamingDataModule(LightningDataModule):
    NUM_ACTIONS = _NUM_ACTIONS
    FRAME_SKIP = _FRAME_SKIP

    def __init__(
        self,
        method: Literal["offline", "online"] = "offline",
        policy: Callable | None = None,
        batch_size: int | None = None,
        batch_traj_len=64,
        num_workers=0,
        max_seen_rtgs: dict[str, np.float64] | None = None,
        pin_memory=torch.cuda.is_available(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["policy"])

        self._batch_size = batch_size
        self.batch_traj_len = batch_traj_len
        self.num_workers = num_workers
        self.num_trajs = batch_size
        self.policy = policy
        self.method: Literal["offline", "online"] = method
        self.pin_memory = pin_memory
        self.max_seen_rtgs = max_seen_rtgs or {}

        # Needed for state loading
        self._start_index = 0
        self._dataset_start_index = 0
        
    def setup_method(self, method: Literal["offline", "online"]):
        self.method = method

    def setup(self, stage):
        # self.stage = stage
        datasets = deepcopy(_DOOM_DATASETS)
        for d in datasets:
            if stage != "fit" or self.method == "online":
                d.policy_maker = lambda: self.policy
            else:
                # TODO: This is whack and only specific to Arnold Models
                d.policy_maker = partial(
                    d.policy_maker, batch_size=self.num_workers or 1
                )

        self._dataset = LazyChainDataset(partial(self._dataset_iterator, datasets))
        self.current_dataset = None

    def _dataset_iterator(
        self, datasets: Iterable[DatasetInfo]
    ) -> Iterator[IterableDataset]:
        random.shuffle(datasets)
        for dataset_info in datasets:
            max_seen_rtg = self.max_seen_rtgs.get(dataset_info.name)
            if max_seen_rtg is not None:
                max_seen_rtg *= dataset_info.target_return_scaling_factor

            create_env_fn = dataset_info.create_env_fn
            create_env_fn = partial(create_env_fn, num_workers=self.num_workers)

            # _dataset_start_index is not 0 if we loaded from a state dict
            size = dataset_info.max_steps - self._dataset_start_index

            torch._dynamo.config.capture_scalar_outputs = True
            # TODO: Make policy random somehow
            policy = dataset_info.policy_maker()
            policy = torch.compile(policy)
            torch._dynamo.config.capture_scalar_outputs = False

            dataset = GymnasiumStreamingDataset(
                size=size,
                batch_size=self.batch_size,
                batch_traj_len=self.batch_traj_len,
                max_traj_len=dataset_info.max_steps_per_traj,
                num_trajs=self.num_trajs,
                policy=policy,
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
            print(f"{self.max_seen_rtgs[dataset_info.name]=}")

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
