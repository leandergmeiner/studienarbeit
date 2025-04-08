from typing import Callable, Final

import torch
from functools import partial
from torchrl.collectors import SyncDataCollector
from data.dataset import DynamicGymnasiumDataset
from data.env import make_env

BATCH_SIZE = 8
BATCH_TRAJ_LEN = 64 # 192 / 3

def get_offline_datasets():
    datasets: Final[list[Callable[..., DynamicGymnasiumDataset]]] = [
        partial(
            DynamicGymnasiumDataset,
            size=100, # TODO
            batch_size=BATCH_SIZE,
            batch_traj_len=BATCH_TRAJ_LEN,
            max_traj_len=1000,
            num_trajs=10, # TODO
            collector_maker=partial(SyncDataCollector, policy=None), # TODO
            env_maker=partial(make_env, "sa/DefendLine-v0"),
        )
    ]
    
    for dataset_maker in datasets:
        yield dataset_maker()

def get_online_datasets(online_policy: Callable):
    online_policy = torch.no_grad(online_policy)
    
    datasets: Final[list[Callable[..., DynamicGymnasiumDataset]]] = [
        partial(
            DynamicGymnasiumDataset,
            size=100, # TODO
            batch_size=BATCH_SIZE,
            batch_traj_len=BATCH_TRAJ_LEN,
            max_traj_len=1000,
            num_trajs=10, # TODO
            collector_maker=partial(SyncDataCollector, policy=online_policy),
            env_maker=partial(make_env, "sa/DefendLine-v0"),
        )
    ]
    
    for dataset_maker in datasets:
        yield dataset_maker()
