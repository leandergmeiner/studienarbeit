# %%
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.probabilistic import InteractionType
from torch.distributions import Bernoulli
from torchrl.modules import (
    ProbabilisticActor,
)
from torchtune.modules import RotaryPositionalEmbeddings
from transformers import AutoModel, GPT2Config, GPT2Model

from modules import DTActor, LightningActor, PatchEmbedding, VideoDT

# from data.doom import get_offline_datasets, get_online_datasets

# from wrappers import AggregateWrapper, VectorAggregateWrapper, Trajectory

# %%
# FIXME: We can not save even 10,000 steps with pixel information, therefore we generate the
# data on the fly. 10,000 steps would take more than 4.3 GB of memory.

base_vit = AutoModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224")

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

transformer = DTActor(
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

# TODO: Reward key: ("next", "reward")
actor = ProbabilisticActor(
    TensorDictModule(
        transformer,
        in_keys=["observation", "action", "return_to_go"],
        out_keys=["logits"],
    ),
    in_keys=["logits"],
    out_keys=["action"],
    distribution_class=Bernoulli,
    default_interaction_type=InteractionType.RANDOM,
)

model = LightningActor(
    actor,
    criterion=torch.nn.CrossEntropyLoss(),
    actor_out_key="logits",
)

# data_module = LightningDataModule.from_datasets(
#     itertools.chain(
#         get_offline_datasets(),
#         get_online_datasets(DecisionTransformerInferenceWrapper(model, inference_context=max_seq_len)),
#     )
# )

# TODO: Use FuseLAMB (Large Batch Optimization for Deep Learning: Training BERT in 76 minutes)
# TODO: Use OnlineDTLoss
# TODO: Tublets, how to report the final reward? Sum should work well.

# TODO: Use Lightning StreamingDataset

# %%
# Test
torch.manual_seed(0)
batch_size = 1
observation = torch.randn((batch_size, 1, 3, 224, 224))
action = torch.randn(
    (batch_size, 0, 12)
)  # FIXME: Can't pass in action size 0. This is necessary to start, though
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

# %%
