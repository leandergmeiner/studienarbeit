# %%
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.dataset import DoomStreamingDataModule
from src.modules import LightningDecisionTransformer

# %%
# This line is needed for some reason to prevent misalignement issues.
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True


# TODO: Maybe use DeiTImageProcesseor
def main():
    model_type = "transformer"
    inference_context = 64

    if model_type == "transformer":
        accumulate_grad_batches = 32
        max_batch_size_in_mem = 2
    elif model_type == "cnn":
        accumulate_grad_batches = 32
        max_batch_size_in_mem = 2

    logger = TensorBoardLogger("logs/", model_type, default_hp_metric=False)
    trainer = Trainer(
        # precision="bf16-true",
        max_epochs=2,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                save_last=True,
                dirpath=f"models/{model_type}",
                every_n_train_steps=200,  # Actually every batch_size // max_batch_size_in_mem iterations
            ),
            # StochasticWeightAveraging(swa_lrs=1e-2),
        ],
    )

    model = LightningDecisionTransformer(
        model_type=model_type,
        num_actions=DoomStreamingDataModule.NUM_ACTIONS,
        inference_context=inference_context,
        target_key="target_action",
        rtg_key="target_return",
        # lr=5e-4,
        lr=5e-3,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    
    datamodule = DoomStreamingDataModule(
        batch_size=max_batch_size_in_mem,
        batch_traj_len=inference_context,
        num_workers=3,
        num_trajs=2,
    )
    
    # Offline training
    datamodule.set_mode("offline", None)
    trainer.fit(model, datamodule=datamodule)
    
    # Online training
    datamodule.set_mode("online", model)
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

# %%
