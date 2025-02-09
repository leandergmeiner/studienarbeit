import torch
from torchtune.modules import RotaryPositionalEmbeddings
from torchrl.modules import (
    ProbabilisticActor,
    DecisionTransformerInferenceWrapper,
)

from tensordict.nn.probabilistic import InteractionType
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.distributions import RelaxedOneHotCategorical
from transformers import DeiTModel, GPT2Config, GPT2Model
from modules import PatchEmbedding, VideoDT, DTActor

from wrappers import AggregateWrapper, VectorAggregateWrapper, Trajectory
from dataloaders.gym import OnlineGymnasiumDataloader

import gymnasium as gym

base_vit = DeiTModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224")

hidden_size = 192

# Aggregate information over 5 frames, this will also have a "smoothing" effect on the models actions
depth = 1
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
        n_layer=6,
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
        model, in_keys=["observation", "action", "return_to_go"], out_keys=["probs"]
    ),
    in_keys=["probs"],
    out_keys=["action"],
    distribution_class=RelaxedOneHotCategorical,
    distribution_kwargs={"temperature": 2.0},  # TODO: Adjust this
    default_interaction_type=InteractionType.RANDOM,
)

# Test
torch.manual_seed(0)
batch_size = 1
observation = torch.randn((batch_size, 1, 3, 224, 224))
action = torch.randn((batch_size, 1, 12)) # FIXME: Can't pass in action size 0. This is necessary to start, though
return_to_go = torch.randn((batch_size, 1, 1))

inputs = TensorDict(
    {"observation": observation, "action": action, "return_to_go": return_to_go},
    batch_size,
)

outputs = actor(inputs)
# x = next_action(x)
print(outputs["action"])

# envs = VectorAggregateWrapper(gym.make_vec("doom"), initial_factory=Trajectory, aggregate=Trajectory.aggregate)
# dataloader = OnlineGymnasiumDataloader(envs, replay_buffer=TODO, return_to_go=TODO, max_ep_len=25_000, max_new_rounds=envs.num_envs)
