# %%
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Callable, Iterable, Iterator, TypeVar

import torch
import torchrl
from tensordict import PersistentTensorDict, TensorDict
from torchrl.collectors import DataCollectorBase
from torchrl.collectors.utils import split_trajectories
from torchrl.data.datasets import BaseDatasetExperienceReplay
from torchrl.data.replay_buffers import (
    ImmutableDatasetWriter,
    LazyMemmapStorage,
    LazyStackStorage,
    LazyTensorStorage,
    ReplayBuffer,
    Sampler,
    SliceSamplerWithoutReplacement,
    TensorDictReplayBuffer,
    Writer,
)
from functools import partial
import numpy as np

from einops import rearrange


def default_observation_transform(
    collector_out_key: str,
    observation_shape: tuple[int, int] = (224, 224),
    reward_key=("next", "reward"),
):
    inverse_transforms = torchrl.envs.Compose(
        # TODO: Naming
        torchrl.envs.Reward2GoTransform(in_keys=reward_key, out_keys=["return_to_go"]),
        torchrl.envs.RenameTransform(
            in_keys=[],
            out_keys=[],
            out_keys_inv=[
                ("original", "pixels"),
                ("next", "original", "pixels"),
            ],
            in_keys_inv=["pixels", ("next", "pixels")],
        ),
    )

    forward_transforms = torchrl.envs.Compose(
        # Forward
        torchrl.envs.ToTensorImage(
            in_keys=["pixels", ("next", "pixels")],
            out_keys=["pixels", ("next", "pixels")],
            shape_tolerant=True,
        ),
        # TODO: Already apply this for .inv(...)
        torchrl.envs.DTypeCastTransform(
            dtype_in=torch.int64,
            dtype_out=torch.float,
            in_keys=["action"],
        ),
        # TODO: Already apply this for .inv(...)
        torchrl.envs.ExcludeTransform(
            "original", ("next", "original"), "labels", "labels_buffer"
        ),
        torchrl.envs.UnaryTransform(
            in_keys=[collector_out_key],
            out_keys=["target_action"],
            fn=lambda td: td.roll(shifts=-1, dims=-2),
        ),
        torchrl.envs.UnaryTransform(
            in_keys=["pixels"],
            out_keys=["pixels"],
            fn=lambda pixels: rearrange(pixels, "... h w -> ... w h"),
        ),
        torchrl.envs.Resize(observation_shape),
    )

    return torchrl.envs.Compose(
        inverse_transforms,
        forward_transforms,
        # TODO: This Rename is awkward
        torchrl.envs.RenameTransform(
            in_keys=[
                "pixels",
                ("next", "pixels"),
            ],
            out_keys=[
                "observation",
                ("next", "observation"),
            ],
        ),
    )


# FIXME: Why is the batch size not constant when iterating???
class GymnasiumStreamingDataset(TensorDictReplayBuffer, torch.utils.data.IterableDataset):
    def __init__(
        self,
        *,
        size: int,
        batch_size: int,
        batch_traj_len: int,
        max_traj_len: int,
        num_trajs: int,
        policy: Callable,
        create_env_fn: Callable,
        collector_maker: Callable,
        collector_out_key: str = "action",
        num_workers: int | None = None,
        storage_maker: Callable = LazyTensorStorage,
        max_seen_rtg: float | None = None,
        make_transform: Callable = default_observation_transform,
        make_transform_kwargs: dict = dict(),
        reward_key=("next", "reward"),
        **kwargs,
    ):
        if any((kwargs.pop("storage", None), kwargs.pop("sampler", None))):
            warnings.warn(
                "`storage` or `sampler` keyword was passed. Their values are ignored."
            )

        assert max_traj_len >= batch_traj_len

        storage_size = num_trajs * max_traj_len

        self.size = size
        self.collector_out_key = collector_out_key

        self.collector_maker = collector_maker
        self.create_env_fn = partial(create_env_fn, max_seen_rtg=max_seen_rtg)
        self.num_workers = num_workers
        self.policy = policy
        self.storage_size = storage_size
        self.max_traj_len = max_traj_len

        self.reward_key = reward_key

        # TODO: Env Batch Size
        storage = storage_maker(
            self.storage_size, ndim=2 if self.num_workers is not None else 1
        )
        sampler = SliceSamplerWithoutReplacement(
            traj_key=("collector", "traj_ids"),
            truncated_key=None,
            slice_len=batch_traj_len,
            strict_length=False,
        )

        transform = make_transform(
            collector_out_key=collector_out_key,
            reward_key=self.reward_key,
            **make_transform_kwargs,
        )

        batch_size_transitions = batch_size * batch_traj_len

        super().__init__(
            batch_size=batch_size_transitions,
            storage=storage,
            sampler=sampler,
            transform=transform,
            **kwargs,
        )

        self._max_seen_rtg = max_seen_rtg or np.finfo(np.float64).min

    def __iter__(self) -> Iterator[TensorDict]:
        collector: DataCollectorBase = self.collector_maker(
            self.create_env_fn,
            create_env_kwargs=dict(num_workers=self.num_workers),
            policy=self.policy,
            total_frames=-1,
            frames_per_batch=self.storage_size,
            max_frames_per_traj=self.max_traj_len,
            reset_when_done=True,
        )

        collect_iterator: Iterator[TensorDict] = collector.iterator()

        num_current_trajs = 0
        while num_current_trajs < self.size:
            self.extend(next(collect_iterator))
            data_iterator = super(TensorDictReplayBuffer, self).__iter__()
            data_iterator = map(split_trajectories, data_iterator)

            for td in data_iterator:
                # [:-1] -> Exclude the time dimension from the #trajectories calculation.
                num_current_trajs += td.batch_size[:-1].numel()

                # Update max seen reward
                self._max_seen_rtg = max(
                    self._max_seen_rtg, np.float64(td[self.reward_key].max())
                )

                yield td

                if num_current_trajs > self.size:
                    break

        collector.shutdown()

    @property
    def max_seen_rtg(self) -> np.float64:
        return self._max_seen_rtg


T = TypeVar("T")


class LazyChainDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets: Iterable[torch.utils.data.Dataset[T]]):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            yield from d


@dataclass
class DataGenSpec:
    collector_maker: Callable
    env_maker: Callable
    name: Path
    total_frames: int


class OfflineGymnasiumDataset(BaseDatasetExperienceReplay):
    def __init__(
        self,
        batch_size: int,
        frames_per_batch: int,
        *,
        specs: Iterable[DataGenSpec] | None = None,
        root: str | Path = "data2/",
        sampler: Sampler | None = None,
        writer: Writer | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: torchrl.envs.Transform | None = None,  # noqa-F821
    ):
        super().__init__(
            storage=LazyStackStorage(),  # We don't want to unnecessarily copy large tensors
            sampler=sampler,
            writer=writer or ImmutableDatasetWriter(),
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            batch_size=batch_size,
            transform=transform,
        )

        self.frames_per_batch = frames_per_batch
        self.specs = specs or self.default_specs
        self.root_dir = Path(root)
        print(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        if not self._is_generated:
            self.generate()

        self._load()

    def generate(self):
        for spec in self.specs:
            print("Generating", spec.name)
            self._generate_from_spec(spec)
            print("Done with")

    def _load(self):
        for path in self.data_paths:
            # Using temporary storage as it should be quite large on Kaggle
            temp_dir = mkdtemp()
            td = PersistentTensorDict.from_h5(path)
            td.memmap(temp_dir)
            self.extend(td)
            shutil.rmtree(temp_dir)

    @property
    def _is_generated(self) -> bool:
        return all(path.exists() for path in self.data_paths)

    def _generate_from_spec(self, spec: DataGenSpec):
        # Using temporary storage as it should be quite large on Kaggle
        with TemporaryDirectory() as temp_dir:
            storage = LazyMemmapStorage(spec.total_frames, scratch_dir=temp_dir)
            buffer = ReplayBuffer(storage=storage)
            collector = spec.collector_maker(
                spec.env_maker,
                total_frames=spec.total_frames,
                frames_per_batch=self.frames_per_batch,
                reset_when_done=True,
                split_trajs=True,
                replay_buffer=buffer,
            )

            for data in collector:
                buffer.extend(data)

            buffer_content: TensorDict = buffer[:]
            buffer_content.to_h5(self.data_path / f"{spec.name}.h5")

    @property
    def data_paths(self):
        return [self.data_path / f"{spec.name}.h5" for spec in self.specs]

    @property
    def data_path(self):
        return self.root_dir

    @property
    def data_path_root(self):
        return self.root_dir

    @property
    def default_specs(self) -> Iterable[DataGenSpec]:
        raise NotImplementedError


# %%
