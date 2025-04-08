# %%
from typing import Callable, Iterable, Iterator
from dataclasses import dataclass
from tempfile import TemporaryDirectory, mkdtemp
import warnings

from pathlib import Path
from tensordict import TensorDict, PersistentTensorDict
import torchrl as rl

from torchrl.data.replay_buffers import (
    ReplayBuffer,
    TensorDictReplayBuffer,
    Sampler,
    SliceSamplerWithoutReplacement,
    LazyStackStorage,
    LazyMemmapStorage,
    LazyTensorStorage,
    Writer,
    ImmutableDatasetWriter,
)
from torchrl.data.datasets import BaseDatasetExperienceReplay
import shutil
from torchrl.collectors.utils import split_trajectories
from torchrl.collectors import DataCollectorBase

from torchrl.envs import Reward2GoTransform

class DynamicGymnasiumDataset(TensorDictReplayBuffer):
    def __init__(
        self,
        *,
        size: int,
        batch_size: int,
        batch_traj_len: int,
        max_traj_len: int,
        num_trajs: int,
        collector_maker: Callable,
        env_maker: Callable,
        storage_maker: Callable = LazyTensorStorage,
        collector_out_key: str = "action",
        **kwargs,
    ):
        if any((kwargs.pop("storage", None), kwargs.pop("sampler", None))):
            warnings.warn(
                "`storage` or `sampler` keyword was passed. Their values are ignored."
            )

        storage = storage_maker(num_trajs * max_traj_len, ndim=1)
        sampler = SliceSamplerWithoutReplacement(
            traj_key=("collector", "traj_ids"),
            truncated_key=None,
            slice_len=batch_traj_len,
            strict_length=False,
        )

        transform = Reward2GoTransform() # We don't need a discount factor

        batch_size_transitions = batch_size * batch_traj_len
        
        super().__init__(
            batch_size=batch_size_transitions,
            storage=storage,
            sampler=sampler,
            transform=transform,
            **kwargs,
        )

        self.collector: DataCollectorBase = collector_maker(
            env_maker,
            total_frames=-1,
            frames_per_batch=max(batch_size_transitions, max_traj_len),
            max_frames_per_traj=max_traj_len,
            reset_when_done=True,
            replay_buffer=self,
        )

        self.size = size
        self.collector_out_key = collector_out_key

    def __iter__(self) -> Iterator[TensorDict]:
        collect_iterator: Iterator[None] = self.collector.iterator()

        i = 0
        while i < self.size:
            next(collect_iterator, None)
            data_iterator = super(TensorDictReplayBuffer, self).__iter__()
            data_iterator = map(split_trajectories, data_iterator)

            for td in data_iterator:
                # td: B x N ...
                td["target"] = td[self.collector_out_key].roll(shifts=-1, dims=1)
                # Zero the last element, as we don't have information for that in td
                td["target"][:, -1] = 0

                yield td

                i += td.shape[0]
                if not i < self.size:
                    break


# %%
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
        transform: rl.envs.Transform | None = None,  # noqa-F821
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
