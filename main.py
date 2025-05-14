# %%
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.tuner import Tuner
from transformers import AutoModel, GPT2Config, GPT2Model

from src.data.doom import (
    DoomOfflineDataModule,
    DoomOnlineDataModule,
    StreamingDataModule,
)
from src.modules import DTActor, LightningSequenceActor, PatchEmbedding, VideoDT

# %%

# FIXME: We can not save even 10,000 steps with pixel information, therefore we generate the
# data on the fly. 10,000 steps would take more than 4.3 GB of memory.

# encoder = AutoModel.from_pretrained(
#     "facebook/deit-tiny-distilled-patch16-224",
#     attn_implementation="sdpa",
#     add_pooling_layer=False,
# )

# TODO
encoder = AutoModel.from_pretrained(
    "microsoft/beit-base-patch16-224",
    attn_implementation="sdpa",
)

hidden_size = encoder.encoder.layer[0].layernorm_after.normalized_shape[0]

# Aggregate information over 4 frames, this will also have a "smoothing" effect on the models actions
depth = 4
base_conv = encoder.embeddings.patch_embeddings.projection
max_patches = 2048
num_spatial_heads = encoder.encoder.layer[0].attention.attention.num_attention_heads
num_temporal_heads = 8

max_seq_len = 64

temporal_transformer = GPT2Model(
    GPT2Config(
        vocab_size=1,
        n_embd=hidden_size,
        n_positions=max_seq_len,
        n_inner=3 * max_seq_len,
        n_layer=48,
        n_head=num_temporal_heads,
        attn_pdrop=0.3,
        embd_pdrop=0.3,
        resid_pdrop=0.3,
        use_flash_attention_2=True,
    )
)

transformer = DTActor(
    VideoDT(
        hidden_size=hidden_size,
        patching=PatchEmbedding(
            base_conv,
            depth,
            encoder.embeddings.patch_embeddings.num_patches,
            "uniform",
            additional_distillation_token=False,
        ),
        num_spatial_heads=num_spatial_heads,
        num_temporal_heads=num_temporal_heads,
        spatial_transformer=encoder.encoder,
        temporal_transformer=temporal_transformer,
    ),
    hidden_dim=hidden_size,
    action_dim=StreamingDataModule.NUM_ACTIONS,
)

# transformer = torch.compile(transformer)

model = LightningSequenceActor(
    transformer,
    criterion=torch.nn.CrossEntropyLoss(),
    labels_key="target_action",
    lr=0.05,
)

# TODO: Use FuseLAMB (Large Batch Optimization for Deep Learning: Training BERT in 76 minutes)
# TODO: Use OnlineDTLoss
# TODO: Tublets, how to report the final reward? Sum should work well.


# %%


# TODO: Maybe use DeiTImageProcesseor
def main():
    # This line is needed for some reason to prevent misalignement issues.
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cudnn.benchmark = True

    batch_size = 32
    max_batch_size_in_mem = 2
    accumulate_grad_batches = batch_size // max_batch_size_in_mem
    print(f"{accumulate_grad_batches=}")

    # TODO: Checkpointing
    trainer = Trainer(
        max_epochs=-1,
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
    )
    # tuner = Tuner(trainer)

    offline_dm = DoomOfflineDataModule(
        batch_size=max_batch_size_in_mem, rounds=1, num_workers=0
    )

    # lr_finder = tuner.lr_find(model, datamodule=offline_dm)
    # print(f"{lr_finder.results=}")

    # tuner.scale_batch_size(model, offline_dm)
    trainer.fit(model, datamodule=offline_dm)

    # online_dm = DoomOnlineDataModule(model, max_seen_rtgs=offline_dm.max_seen_rtgs)
    # trainer.fit(model, datamodule=online_dm)


if __name__ == "__main__":
    main()

# %%
