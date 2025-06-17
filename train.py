# %%
from pathlib import Path

import fire
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.dataset import DoomStreamingDataModule
from src.modules import LightningDecisionTransformer

# %%
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True


def main(model_type="transformer", inference_context=64):
    if model_type == "transformer":
        accumulate_grad_batches = 32
        max_batch_size_in_mem = 2
    elif model_type == "cnn":
        accumulate_grad_batches = 32
        max_batch_size_in_mem = 2

    logger = TensorBoardLogger(
        "rsrc", model_type, sub_dir="logs", default_hp_metric=False
    )

    model_checkpoint = ModelCheckpoint(
        save_last=True,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        # 2* since we call .step of the two optimizers
        every_n_train_steps=1000 * 2,
        dirpath=Path(logger.root_dir) / "models",
    )

    trainer = Trainer(
        max_epochs=3,
        log_every_n_steps=50,
        val_check_interval=2000,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[
            model_checkpoint,
            # StochasticWeightAveraging(swa_lrs=1e-2),
        ],
    )

    model = LightningDecisionTransformer(
        model_type=model_type,
        num_actions=DoomStreamingDataModule.NUM_ACTIONS,
        inference_context=inference_context,
        target_key="target_action",
        rtg_key="target_return",
        lr=5e-4,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # Offline training
    print("Offline Training")
    datamodule = DoomStreamingDataModule(
        policy=model,
        batch_size=max_batch_size_in_mem,
        batch_traj_len=inference_context,
        num_workers=3,
    )
    
    datamodule.set_mode("offline", None)
    trainer.fit(model, datamodule=datamodule, ckpt_path="last")

    # Online training
    print("Online Training")
    datamodule = DoomStreamingDataModule(
        policy=model,
        batch_size=max_batch_size_in_mem,
        batch_traj_len=inference_context,
        max_seen_rtgs=datamodule.max_seen_rtgs,
        num_workers=0, # TODO: The need of setting this to 0 is whack
    )

    # Update the number of max epochs to include the online training
    trainer.fit_loop.max_epochs = 2 * trainer.fit_loop.max_epochs
    datamodule.set_mode("online", model)
    trainer.fit(model, datamodule=datamodule, ckpt_path="last")


if __name__ == "__main__":
    fire.Fire(main)
