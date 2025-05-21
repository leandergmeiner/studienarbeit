import warnings
from functools import partial
from itertools import islice
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
import torchrl.envs
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import (
    LazyTensorStorage,
    SliceSampler,
    TensorDictReplayBuffer,
)
from torchrl.collectors import SyncDataCollector


def default_observation_transform(
    collector_out_key: str,
    observation_shape: tuple[int, int] = (224, 224),
    reward_key=("next", "reward"),
    exclude_next_observation: bool = False,
):
    pixels_keys = (
        ["pixels", ("next", "pixels")] if exclude_next_observation else ["pixels"]
    )

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
        torchrl.envs.ExcludeTransform("labels", "labels_buffer", inverse=True),
        torchrl.envs.DTypeCastTransform(
            dtype_in=torch.int64,
            dtype_out=torch.bool,
            in_keys_inv=["action"],
        ),
    )

    if exclude_next_observation:
        inverse_transforms.append(
            torchrl.envs.ExcludeTransform(("next", "pixels"), inverse=True)
        )

    forward_transforms = torchrl.envs.Compose(
        # Forward
        torchrl.envs.ToTensorImage(
            in_keys=pixels_keys,
            out_keys=pixels_keys,
            shape_tolerant=True,
        ),
        torchrl.envs.ExcludeTransform("original", ("next", "original")),
        # TODO: Already apply this for .inv(...)
        torchrl.envs.DTypeCastTransform(
            dtype_in=torch.bool,
            dtype_out=torch.float,
            in_keys=["action"],
        ),
        # TODO: Already apply this for .inv(...)
        torchrl.envs.UnaryTransform(
            in_keys=[collector_out_key],
            out_keys=["target_action"],
            fn=lambda td: td.roll(shifts=-1, dims=-2),
        ),
        torchrl.envs.UnaryTransform(
            in_keys=pixels_keys,
            out_keys=pixels_keys,
            fn=lambda pixels: rearrange(pixels, "... h w -> ... w h"),
        ),
        torchrl.envs.Resize(observation_shape, in_keys=pixels_keys),
    )

    # TODO: This Rename is awkward
    if exclude_next_observation:
        forward_transforms.append(
            torchrl.envs.RenameTransform(
                in_keys=["pixels"],
                out_keys=["observation"],
            ),
        )
    else:
        forward_transforms.append(
            torchrl.envs.RenameTransform(
                in_keys=[
                    "pixels",
                    ("next", "pixels"),
                ],
                out_keys=[
                    "observation",
                    ("next", "observation"),
                ],
            )
        )

    return torchrl.envs.Compose(
        inverse_transforms,
        forward_transforms,
    )


# FIXME: Why is the batch size not constant when iterating???
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
        policy: Callable,
        create_env_fn: Callable | torchrl.envs.EnvCreator,
        collector_out_key: str = "action",
        num_workers: int | None = None,
        max_seen_rtg: float | None = None,
        make_transform: Callable = default_observation_transform,
        make_transform_kwargs: dict = dict(),
        reward_key=("next", "reward"),
        compilable: bool = True,
        **kwargs,
    ):
        if any((kwargs.pop("storage", None), kwargs.pop("sampler", None))):
            warnings.warn(
                "`storage` or `sampler` keyword was passed. Their values are ignored."
            )

        if batch_size > num_trajs:
            warnings.warn(
                f"Can not utilize full batch size ({batch_size}), as only {num_trajs} are kept in memory at one time."
            )

        assert max_traj_len >= batch_traj_len

        storage_size = num_trajs * max_traj_len

        self.batch_traj_len = batch_traj_len
        self.num_slices = batch_size
        self.max_steps = size
        self.collector_out_key = collector_out_key

        self.create_env_fn = partial(create_env_fn, max_seen_rtg=max_seen_rtg)
        self.num_workers = num_workers
        self.policy = policy
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

        # sampler = SliceSamplerWithoutReplacement(
        #     # end_key=("next", "done"),
        #     traj_key=("collector", "traj_ids"),
        #     truncated_key=None,
        #     slice_len=batch_traj_len,
        #     strict_length=False,
        #     compile=True,
        #     use_gpu=True,
        # )
        # TODO: SliceSamplerWithoutReplacement seems to not work correctly
        # It only samples a slice once from a trajectory, instead of multiple
        # non-equal slices from the same trajectory.
        sampler = SliceSampler(
            # FIXME
            # traj_key=("collector", "traj_ids"),
            end_key=("next", "done"),  # TODO
            slice_len=batch_traj_len,
            # strict_length=False,
            cache_values=True,
            compile=dict(fullgraph=True, mode="reduce-overhead")
            if self.compilable
            else False,
            use_gpu=self.compilable,
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
            shared=True,
            compilable=True,
            **kwargs,
        )

        self._max_seen_rtg = max_seen_rtg

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
                    break

        collector.shutdown()

    def collector(self):
        return SyncDataCollector(
            self.create_env_fn,
            create_env_kwargs=dict(num_workers=self.num_workers),
            policy=self.policy,
            total_frames=-1,
            frames_per_batch=self.storage_size,
            max_frames_per_traj=self.max_traj_len,
            reset_when_done=True,
            split_trajs=True,
            # We fill the replay buffer with one batch completely
            # therefore we can always reuse the same tensordict
            return_same_td=True,
            compile_policy=self.compilable,
        )

    @property
    def max_seen_rtg(self) -> np.float64 | None:
        return self._max_seen_rtg

    def __len__(self) -> int:
        return self.max_steps // (self.num_slices * self.batch_traj_len)


class LazyChainDataset(torch.utils.data.IterableDataset):
    def __init__(self, make_datasets: Callable[[], Iterable]):
        super().__init__()
        self.make_datasets = make_datasets

    def __iter__(self):
        for d in self.make_datasets():
            yield from d

    # def __len__(self):
    #     s = sum(len(d) for d in self.make_datasets())
    #     return s
