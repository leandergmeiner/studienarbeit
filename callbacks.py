import warnings
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from tensordict import TensorDict
import torchrl.data
from wrappers import EpisodeWrapper, Episode
from dataloaders.gym import GymnasiumDataloader


class GymnasiumCallback(Callback):
    def __init__(
        self,
        num_generated_runs_per_epoch: int,
        apply_action_for_n_frames: int,
        out_key="pred",
    ):
        self.apply_action_for_n_frames = apply_action_for_n_frames
        self.num_generated_runs_per_epoch = num_generated_runs_per_epoch
        self.out_key = out_key

    def on_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, dataloader: Any
    ):
        if not isinstance(dataloader, GymnasiumDataloader):
            return

        # Play N games and sample their trajectories
        dataloader.envs.reset()

        # TODO: This is whack
        with torch.no_grad():
            while (
                dataloader.envs.num_finished_episodes
                < self.num_generated_runs_per_epoch
            ):
                inputs = _episodes_to_tensordict(dataloader.envs.episodes)
                # FIXME: Change this to pl_module.generate to take advantage of generation speedups
                next_action = pl_module(inputs)[self.out_key]
                
                #FIXME: Apply action for N frames
                for _ in range(self.apply_action_for_n_frames):
                    dataloader.envs.step(next_action)

        dataloader.envs.reset()

        # New runs are now in the replay buffer of the dataloader

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        return self.on_epoch_start(trainer, pl_module, pl_module.train_dataloader())
