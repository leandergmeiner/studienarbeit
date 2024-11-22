import torch
from torchtune.modules import RotaryPositionalEmbeddings
from torchrl.modules import (
    ProbabilisticActor,
    DecisionTransformerInferenceWrapper,
    TanhDelta,
)
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from transformers import DeiTModel, GPT2Config, GPT2Model
from modules import PatchEmbedding, VideoDT, DTActor

base_vit = DeiTModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224")

hidden_size = 192

# Aggregate information over 5 frames, this will also have a "smoothing" effect on the models actions
depth = 5
base_conv = base_vit.embeddings.patch_embeddings.projection
max_patches = 2048
num_spatial_heads = 8
num_temporal_heads = 8

max_seq_len = 200

num_actions = 12

# TODO: Convert to class
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

model = DTActor(
    VideoDT(
        hidden_size=hidden_size,
        patching=PatchEmbedding(base_conv, depth),
        # TODO: Read RotaryPositionalEmbeddings paper
        # TODO: Calculate max_patches based on max_seq_len
        pos_embedding=RotaryPositionalEmbeddings(
            hidden_size // num_spatial_heads,
            max_patches,
        ),
        num_spatial_heads=num_spatial_heads,
        num_temporal_heads=num_temporal_heads,
        spatial_transformer=base_vit.encoder,
        temporal_transformer=temporal_transformer,
    ),
    hidden_dim=hidden_size,
    action_dim=num_actions,
)

actor = ProbabilisticActor(
    TensorDictModule(
        model, in_keys=["observation", "action", "return_to_go"], out_keys=["param"]
    ),
    in_keys=["param"],
    out_keys=["action"],
    distribution_class=TanhDelta,  # TODO: What is TanhDelta?
    distribution_kwargs={"low": -1.0, "high": 1.0},
)

# Test
observation = torch.randn((1, 20, 3, 224, 224))
action = torch.randn((1, 4, 12))
return_to_go = torch.randn((1, 4, 1))

inputs = TensorDict(
    {"observation": observation, "action": action, "return_to_go": return_to_go}
)

outputs = actor(inputs)
# x = next_action(x)
print(outputs)
