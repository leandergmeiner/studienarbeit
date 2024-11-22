import torch
import torch._dynamo
from torchtune.modules import (
    TransformerDecoder,
    TransformerSelfAttentionLayer,
    MultiHeadAttention,
    RotaryPositionalEmbeddings,
)

from transformers import DeiTModel
from modules import PatchEmbedding, VideoDT, get_simple_temporal_decoder

base_vit = DeiTModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224")

hidden_size = 192

# Aggregate information over 5 frames, this will also have a "smoothing" effect on the models actions
depth = 5
base_conv = base_vit.embeddings.patch_embeddings.projection
max_patches = 1024
num_spatial_heads = 8
num_temporal_heads = 8

max_seq_len = 200

temporal_transformer = get_simple_temporal_decoder(
    hidden_size, num_temporal_heads, max_seq_len, 4
)

model = VideoDT(
    hidden_size=hidden_size,
    patching=PatchEmbedding(base_conv, depth),
    # TODO: Read RotaryPositionalEmbeddings paper
    pos_embedding=RotaryPositionalEmbeddings(
        hidden_size // num_spatial_heads,
        max_patches,
    ),
    num_spatial_heads=num_spatial_heads,
    num_temporal_heads=num_temporal_heads,
    spatial_transformer=base_vit.encoder,
    temporal_transformer=temporal_transformer
)

# Test
frames = torch.randn((1, 20, 3, 224, 224))
actions = torch.randn((1, 4, 12))
return_to_go = torch.randn((1, 4, 1))

x = model(frames, actions, return_to_go, mask=torch.ones(4, 4))