from typing import Callable, Protocol

import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
import torchrl.data
from wrappers import VectorAggregateWrapper, Trajectory

class OnlineGymnasiumDataloader:
    def __init__(self, envs: VectorAggregateWrapper[Trajectory], replay_buffer: torchrl.data.ReplayBuffer, return_to_go: float, max_ep_len: int, max_new_rounds: int, out_key: str = "action"):
        self.envs = envs
        self.replay_buffer = replay_buffer
        self.return_to_go = return_to_go
        self.max_ep_len = max_ep_len
        self.max_new_rounds = max_new_rounds
        self.out_key = out_key
                
    def update(self, policy: TensorDictModule):
        policy = torch.no_grad(policy)
        device = policy.device
        
        # Values are already aggregated by the VectorAggregateWrapper
        _ = self.envs.reset()
        
        for _ in range(self.max_ep_len):
            # All finished
            if len(self.envs.aggregations) >= self.max_new_rounds:
                break
        
            inputs = TensorDict([c.to_tensordict() for c in self.envs.current], device=device)
                        
            actions: torch.Tensor = policy(inputs)[self.out_key]
            actions = actions.detach().cpu().numpy()
            
            # Values are already aggregated by the VectorAggregateWrapper
            _ = self.envs.step(actions)
        
    def __iter__(self):
        self.replay_buffer.extend(self.envs.aggregations[:self.max_new_rounds])
        self.envs.reset() # Clear self.envs.aggregations
        return iter(self.replay_buffer)
        
        