from typing import Callable

import torch
from tensordict import TensorDict
import torchrl.data
from wrappers import VectorAggregateWrapper

class GymnasiumDataloader:
    def __init__(self, envs: VectorAggregateWrapper, replay_buffer: torchrl.data.ReplayBuffer, rounds: int):
        self.envs = envs
        self.replay_buffer = replay_buffer
        self.rounds = rounds
        
    def play(self, policy: Callable[[TensorDict], TensorDict]):
        self.envs.reset()
        
        policy = torch.no_grad(policy)
        
        while len(self.envs.aggregations) < self.rounds:
            
        
        
    def __iter__(self):
        self.envs.reset()
        return iter(self.replay_buffer)
        
        