from typing import Callable, Final

import torch
from functools import partial
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from src.data.dataset import DynamicGymnasiumDataset
from src.data.env import make_env

from src.data.transforms import SaveOriginalValuesTransform
from torchrl.envs import Compose, EnvBase, TransformedEnv

from pretrained.models import ArnoldAgent

from torchrl.data.replay_buffers import LazyTensorStorage

BATCH_SIZE = 8
BATCH_TRAJ_LEN = 64 # 192 / 3
NUM_TRAJS = 2 # TODO

def get_offline_datasets():
    datasets: Final[list[Callable[..., DynamicGymnasiumDataset]]] = [
        partial(
            DynamicGymnasiumDataset,
            size=100, # TODO
            batch_size=BATCH_SIZE,
            batch_traj_len=BATCH_TRAJ_LEN,
            max_traj_len=1000,
            num_trajs=NUM_TRAJS,
            policy=ArnoldAgent(),
            collector_maker=SyncDataCollector, # TODO
            num_workers=2,
            create_env_fn=partial(make_env, "arnold/DefendCenter-v0"),
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
            create_env_fn=partial(make_env, "sa/DefendLine-v0"),
        )
    ]
    
    for dataset_maker in datasets:
        yield dataset_maker()
