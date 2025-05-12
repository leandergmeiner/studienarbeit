# %%
import torch
from lightning import Trainer
from transformers import AutoModel, GPT2Config, GPT2Model

from src.data.doom import DoomOfflineDataModule
from src.modules import DTActor, LightningSequenceActor, PatchEmbedding, VideoDT

# %%
# This line is needed for some reason to prevent misalignement issues.
torch.backends.cuda.enable_mem_efficient_sdp(False)

# FIXME: We can not save even 10,000 steps with pixel information, therefore we generate the
# data on the fly. 10,000 steps would take more than 4.3 GB of memory.

base_vit = AutoModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224")

hidden_size = base_vit.encoder.layer[0].layernorm_after.normalized_shape[0]

# Aggregate information over 4 frames, this will also have a "smoothing" effect on the models actions
depth = 4
base_conv = base_vit.embeddings.patch_embeddings.projection
max_patches = 2048
num_spatial_heads = base_vit.encoder.layer[0].attention.attention.num_attention_heads
num_temporal_heads = 8

max_seq_len = 64 * 3

num_actions = 10 # TODO

temporal_transformer = GPT2Model(
    GPT2Config(
        vocab_size=1,
        n_embd=hidden_size,
        n_positions=max_seq_len,
        n_inner=max_seq_len,
        n_layer=12,
        n_head=num_temporal_heads,
    )
)

transformer = DTActor(
    VideoDT(
        hidden_size=hidden_size,
        patching=PatchEmbedding(
            base_conv, depth, base_vit.embeddings.patch_embeddings.num_patches, "uniform"
        ),
        # TODO: Read RotaryPositionalEmbeddings paper
        # TODO: Calculate max_patches based on max_seq_len
        num_spatial_heads=num_spatial_heads,
        num_temporal_heads=num_temporal_heads,
        spatial_transformer=base_vit.encoder,
        temporal_transformer=temporal_transformer,
    ),
    hidden_dim=hidden_size,
    action_dim=num_actions,
)

model = LightningSequenceActor(
    transformer,
    criterion=torch.nn.CrossEntropyLoss(),
    labels_key="target_action",
)

# TODO: Use FuseLAMB (Large Batch Optimization for Deep Learning: Training BERT in 76 minutes)
# TODO: Use OnlineDTLoss
# TODO: Tublets, how to report the final reward? Sum should work well.

# TODO: Use Lightning StreamingDataset

# %%

# TODO: Maybe use DeiTImageProcesseor

def main():
    trainer = Trainer(max_epochs=-1, log_every_n_steps=20)
    trainer.fit(model, datamodule=DoomOfflineDataModule(rounds=1, batch_size=2))


if __name__ == "__main__":
    main()

# %%
