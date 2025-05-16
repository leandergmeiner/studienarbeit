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
    method = "transformer"
    
    model = LightningSequenceActor.default(
        method=method,
        frame_skip=DoomStreamingDataModule.FRAME_SKIP,
        num_actions=DoomStreamingDataModule.NUM_ACTIONS,
        inference_context=64,
        criterion=torch.nn.CrossEntropyLoss(),
        labels_key="target_action",
        lr=0.05,
    )

    batch_size = 64
    max_batch_size_in_mem = 1
    accumulate_grad_batches = batch_size // max_batch_size_in_mem

    logger = TensorBoardLogger("logs/", f"dt-{method}", default_hp_metric=False)
    trainer = Trainer(
        max_epochs=-1,
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger,
        callbacks=[
            # TODO:
            ModelCheckpoint(
                save_top_k=10,
                monitor="global_step",
                mode="max",
                dirpath="models",
                every_n_train_steps=100,  # Actually every batch_size // max_batch_size_in_mem iterations
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
