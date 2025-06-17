import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable, Iterator, Literal

import dill  # noqa: F401
import numpy as np
import torch
import vizdoom as vzd
from lightning import LightningDataModule
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.utils.data import DataLoader
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
    method: Literal["offline", "online"] = "offline"
    policy_maker: torch.nn.Module | None = None
    make_env_transforms: Callable[..., list] = list
    make_dataset_transforms: Callable[..., list] = list
    target_return_scaling_factor: float = 1.5
    guessed_target_return: float | None = None
    rtg_key: str = "target_return"


_NUM_ACTIONS = 10
# _FRAME_SKIP = 4  # TODO: This number is not enforced


def make_env_serial(dataset_info: DatasetInfo, max_seen_rtg: float, num_envs: int = 1):
    def make_env():
        env = dataset_info.create_env_fn()
        wrapped = TransformedEnv(
            env,
            Compose(*dataset_info.make_env_transforms(target_return=max_seen_rtg)),
        )
        return wrapped

    return SerialEnv(num_envs, make_env)


class DoomStreamingDataModule(LightningDataModule):
    NUM_ACTIONS = _NUM_ACTIONS
    # FRAME_SKIP = _FRAME_SKIP

    def __init__(
        self,
        policy: TensorDictModule,
        batch_size: int,
        batch_traj_len: int = 64,
        method: Literal["offline", "online"] = "offline",
        num_workers: int | None = None,
        num_trajs: int = 0,
        max_seen_rtgs: dict[str, np.float64] | None = None,
        pin_memory=torch.cuda.is_available(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["policy", "pin_memory"])

        self.batch_size = batch_size
        self.batch_traj_len = batch_traj_len
        self.num_trajs = num_trajs or batch_size
        self.num_workers = num_workers if num_workers is not None else self.num_trajs
        self.policy = policy
        self.method = method
        self.pin_memory = pin_memory
        self.max_seen_rtgs = max_seen_rtgs or {}

        # Needed for state loading
        self._start_index = 0
        self._dataset_start_index = 0

        self._device = torch.device("cuda:0" if self.pin_memory else "cpu")

    def set_mode(
        self, method: Literal["offline", "online"], policy: TensorDictModule | None
    ):
        if method == "online":
            assert policy is not None

        self.method = method
        self.policy = policy

        if self.policy is not None and hasattr(self.policy, "method"):
            policy.method = method

    def setup(self, stage):
        if stage == "fit":
            datasets = self._get_datasets()
            total_length = sum(
                dataset.max_steps // (self.batch_size * self.batch_traj_len)
                for dataset in datasets
            )

            self._make_validation_datasets = partial(
                self._dataset_iterator,
                self._get_datasets(method="online"),
                method="online",
            )

            self._train_dataset = LazyChainDataset(
                partial(self._dataset_iterator, datasets), total_length=total_length
            )
        else:
            self._make_validation_datasets = None
            self._train_dataset = None

    def _get_datasets(
        self, method: Literal["offline", "online"] | None = None
    ) -> list[DatasetInfo]:
        method = method or self.method

        datasets = []
        for d in self.available_datasets():
            if d.method != method:
                continue

            if isinstance(d.policy_maker, partial):
                func = d.policy_maker.func
            else:
                func = d.policy_maker

            if func is not None and "batch_size" in inspect.getfullargspec(func).args:
                d.policy_maker = partial(d.policy_maker, batch_size=1)

            datasets.append(d)

        return datasets

    def _dataset_iterator(
        self,
        datasets: Iterable[DatasetInfo],
        method: Literal["offline", "online"] | None = None,
    ) -> Iterator[GymnasiumStreamingDataset]:
        method = method or self.method

        for dataset_info in datasets:
            max_seen_rtg = self.max_seen_rtgs.get(dataset_info.name)
            if max_seen_rtg is not None:
                max_seen_rtg *= dataset_info.target_return_scaling_factor

            if (
                method == "online"
                and max_seen_rtg is None
                and dataset_info.guessed_target_return is not None
            ):
                max_seen_rtg = dataset_info.guessed_target_return

            # _dataset_start_index is not 0 if we loaded from a state dict
            size = dataset_info.max_steps - self._dataset_start_index

            def online_policy():
                return self.policy

            if method != "online" and dataset_info.policy_maker:
                policy = dataset_info.policy_maker()
                policy = torch.compile(policy)
            else:
                policy = online_policy()

            # assert policy is not None

            dataset = GymnasiumStreamingDataset(
                size=size,
                batch_size=self.batch_size,
                batch_traj_len=self.batch_traj_len,
                max_traj_len=dataset_info.max_steps_per_traj,
                num_trajs=self.num_trajs,
                alpha=1.0,
                beta=1.0,
                priority_key=dataset_info.rtg_key,
                policy=policy,
                max_seen_rtg=max_seen_rtg,
                create_env_fn=partial(make_env_serial, dataset_info, max_seen_rtg),
                transform=Compose(*dataset_info.make_dataset_transforms()),
                compilable=True,
            )

            self.current_dataset = dataset

            yield dataset

            if hasattr(self.policy, "reset") and callable(self.policy.reset):
                self.policy.reset()

            self._start_index += 1

            self.max_seen_rtgs[dataset_info.name] = dataset.max_seen_rtg

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            collate_fn=torch.cat,
            in_order=False,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            pin_memory=self.pin_memory,
            pin_memory_device="cuda:0" if self.pin_memory else None,
        )

    # def _val_dataset_iterator(self) -> Iterator[TensorDict]:
    #     for dataset in self._make_validation_datasets():
    #         td = next(dataset.collector().iterator())
    #         del td["observation"]  # TODO: This is whack
    #         del td[("next", "observation")]  # TODO: This is whack
    #         yield td

    # def val_dataloader(self):
    #     return DataLoader(
    #         LazyChainDataset(self._val_dataset_iterator),
    #         collate_fn=torch.cat,
    #         in_order=False,
    #         pin_memory=self.pin_memory,
    #         pin_memory_device="cuda:0" if self.pin_memory else None,
    #     )

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

    @staticmethod
    def available_datasets():
        return [
            DatasetInfo(
                name="defend_the_center",
                method="offline",
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
                rtg_key="target_return",
            ),
            DatasetInfo(
                name="defend_the_center",
                method="online",
                policy_maker=None,  # Can be none, since it's online and the policy is replaced.
                create_env_fn=partial(GymEnv, "sa/ArnoldDefendCenter-v0"),
                make_env_transforms=partial(
                    online_env_make_transforms, observation_shape=(224, 224)
                ),
                make_dataset_transforms=partial(
                    online_dataset_make_transforms,
                    rtg_key="target_return",
                    collector_out_key="action",
                ),
                max_steps=250_000,
                max_steps_per_traj=500,
                guessed_target_return=150.0,
                rtg_key="target_return",
            ),
            # DatasetInfo(
            #     name="health_gathering",
            #     method="offline",
            #     policy_maker=partial(ArnoldAgent, "health_gathering"),
            #     create_env_fn=partial(GymEnv, "sa/ArnoldHealthGathering-v0"),
            #     make_env_transforms=partial(
            #         arnold_env_make_transforms,
            #         game_variables=[vzd.GameVariable.HEALTH],
            #         frame_skip=4,
            #     ),
            #     make_dataset_transforms=partial(
            #         arnold_dataset_make_transforms,
            #         observation_shape=(224, 224),
            #         exclude_next_observation=True,
            #         collector_out_key="action",
            #         rtg_key="target_return",
            #     ),
            #     max_steps=1_000_000,
            #     max_steps_per_traj=500,
            # ),
            # DatasetInfo(
            #     name="health_gathering",
            #     method="online",
            #     policy_maker=None,  # Can be none, since it's online and the policy is replaced.
            #     create_env_fn=partial(GymEnv, "sa/ArnoldHealthGathering-v0"),
            #     make_env_transforms=partial(
            #         online_env_make_transforms, observation_shape=(224, 224)
            #     ),
            #     make_dataset_transforms=partial(
            #         online_dataset_make_transforms,
            #         rtg_key="target_return",
            #         collector_out_key="action",
            #     ),
            #     max_steps=250_000,
            #     max_steps_per_traj=500,
            #     guessed_target_return=30.0,
            # ),
        ]
