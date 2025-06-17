from itertools import islice
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
import torchrl.envs
from tensordict import TensorDict
from tensordict.nn.probabilistic import InteractionType
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSliceSampler,
    TensorDictReplayBuffer,
)


class GymnasiumStreamingDataset(
    TensorDictReplayBuffer, torch.utils.data.IterableDataset
):
    def __init__(
        self,
        *,
        size: int,
        batch_size: int,
        batch_traj_len: int,
        max_traj_len: int,
        num_trajs: int,
        alpha: float,
        beta: float,
        policy: torch.nn.Module | None,
        create_env_fn: Callable | torchrl.envs.EnvCreator,
        reward_key=("next", "reward"),
        compilable: bool = True,
        transform: torchrl.envs.Transform | None = None,
        max_seen_rtg: float | None = None,
        priority_key: str | None = None,
    ):
        assert max_traj_len >= batch_traj_len

        storage_size = num_trajs * max_traj_len

        self.batch_traj_len = batch_traj_len
        self.num_slices = batch_size
        self.max_steps = size
        self.create_env_fn = create_env_fn
        self.policy = policy
        self._max_seen_rtg = max_seen_rtg
        self.storage_size = storage_size
        self.max_traj_len = max_traj_len
        self.reward_key = reward_key
        self.compilable = compilable

        storage = LazyTensorStorage(
            # FIXME:
            # num_trajs, ndim=1
            # self.storage_size, ndim=1
            self.storage_size,
            ndim=1,
            compilable=self.compilable,
        )

        compile_kwargs = (
            dict(fullgraph=True, mode="reduce-overhead") if self.compilable else False
        )

        # SliceSamplerWithoutReplacement seems to not work correctly
        # It only samples a slice once from a trajectory, instead of multiple
        # non-equal slices from the same trajectory.
        sampler = PrioritizedSliceSampler(
            max_capacity=storage_size,
            alpha=alpha,
            beta=beta,
            end_key=("next", "done"),
            slice_len=batch_traj_len,
            # strict_length=False,
            cache_values=True,
            compile=compile_kwargs,
            use_gpu=self.compilable,
        )
        
        batch_size_transitions = batch_size * batch_traj_len

        super().__init__(
            batch_size=batch_size_transitions,
            storage=storage,
            sampler=sampler,
            transform=transform,
            shared=True,
            compilable=True,
            priority_key=priority_key or self.reward_key,
        )

    def __iter__(self) -> Iterator[TensorDict]:
        collector = self.collector()
        collect_iterator: Iterator[TensorDict] = collector.iterator()

        num_steps = 0
        while num_steps < self.max_steps:
            td = next(collect_iterator)
            td = td.flatten(0, 1)
            self.extend(td)
            data_iterator = super(TensorDictReplayBuffer, self).__iter__()
            data_iterator: Iterator[TensorDict] = islice(
                data_iterator, self.max_traj_len // 4
            )  # TODO

            for td in data_iterator:
                td = td.reshape(self.num_slices, -1)

                # [:-1] -> Exclude the time dimension from the num of trajectories calculation.
                num_steps += td.batch_size.numel()

                # Update max seen reward
                if self._max_seen_rtg is not None:
                    self._max_seen_rtg = max(
                        self._max_seen_rtg, np.float64(td[self.reward_key].max())
                    )
                else:
                    self._max_seen_rtg = np.float64(td[self.reward_key].max())

                yield td

                if num_steps > self.max_steps:
                    return

        collector.shutdown()

    def collector(self):
        if (
            self.policy is not None
            and hasattr(self.policy, "reset")
            and callable(self.policy.reset)
        ):
            self.policy.reset()

        return SyncDataCollector(
            self.create_env_fn,
            policy=self.policy,
            total_frames=-1,
            frames_per_batch=self.storage_size,
            max_frames_per_traj=self.max_traj_len,
            reset_when_done=True,
            split_trajs=True,
            # We fill the replay buffer with one batch completely
            # therefore we can always reuse the same tensordict
            return_same_td=True,
            # compile_policy=self.compilable,
            # device="cpu",
            exploration_type=InteractionType.RANDOM,
        )

    @property
    def max_seen_rtg(self) -> np.float64 | None:
        return self._max_seen_rtg

    def __len__(self) -> int:
        return self.max_steps // (self.num_slices * self.batch_traj_len)


class LazyChainDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, make_datasets: Callable[[], Iterable], total_length: int | None = None
    ):
        super().__init__()
        self.make_datasets = make_datasets
        self._len = total_length

    def __iter__(self):
        for d in self.make_datasets():
            yield from d

    def __len__(self):
        # s = sum(len(d) for d in self.make_datasets())
        # return s
        return self._len
