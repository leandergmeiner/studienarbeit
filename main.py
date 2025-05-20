# %%
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.doom import DoomStreamingDataModule
from src.modules import LightningSequenceActor

# %%

# TODO: Use FuseLAMB (Large Batch Optimization for Deep Learning: Training BERT in 76 minutes)
# TODO: Use OnlineDTLoss
# TODO: Tublets, how to report the final reward? Sum should work well.

# This line is needed for some reason to prevent misalignement issues.
torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cudnn.benchmark = True


# TODO: Maybe use DeiTImageProcesseor
def main():
    method = "cnn"

    model = LightningSequenceActor(
        method=method,
        frame_skip=DoomStreamingDataModule.FRAME_SKIP,
        num_actions=DoomStreamingDataModule.NUM_ACTIONS,
        inference_context=64,
        labels_key="target_action",
        lr=0.005,
    )

    if method == "transformer":
        accumulate_grad_batches = 16
        max_batch_size_in_mem = 3
    elif method == "cnn":
        accumulate_grad_batches = 24
        max_batch_size_in_mem = 2

    logger = TensorBoardLogger("logs/", f"dt-{method}", default_hp_metric=False)
    trainer = Trainer(
        precision="transformer-engine-float16",  # If H100 is available
        max_epochs=-1,
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                save_last=True,
                dirpath=f"models/{method}",
                every_n_train_steps=10,  # Actually every batch_size // max_batch_size_in_mem iterations
            ),
            StochasticWeightAveraging(swa_lrs=1e-2),
        ],
    )
    # tuner = Tuner(trainer)

    datamodule = DoomStreamingDataModule(
        policy=model,
        batch_size=max_batch_size_in_mem,
        num_workers=1,
    )
    datamodule.setup("offline")
    trainer.fit(model, datamodule=datamodule)

    # online_dm = DoomOnlineDataModule(model, max_seen_rtgs=offline_dm.max_seen_rtgs)
    # trainer.fit(model, datamodule=online_dm)


if __name__ == "__main__":
    main()

# %%
