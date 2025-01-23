import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Final

import gymnasium as gym
import numpy as np
import torch
import torchrl.data
from tensordict import TensorDict


@dataclass
class Episode:
    observations: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    returns_to_go: list[float] = field(default_factory=list)
    rewards: list[float] = field(
        default_factory=list
    )  # Used for hindsight return labeling

    max_return_to_go: float = 0.0
    has_stopped: bool = False

    def step(self, action, observation: Any, reward: float, has_stopped: bool):
        self.has_stopped = has_stopped
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

        self.max_return_to_go -= reward
        self.returns_to_go.append(self.max_return_to_go)

    def hindsight_label(self):
        """Perform hindsight labeling from the Online Decision Transformer paper."""
        returns_to_go = np.cumsum(self.rewards[::-1])[::-1]
        self.returns_to_go = list(returns_to_go)

    def to_tensordict(self):
        td = TensorDict(
            {
                "observation": [self.observations],
                "action": [self.actions],
                "return_to_go": [self.returns_to_go],
                "reward": [self.rewards],
            },
            batch_size=1,
        )

        return td


class EpisodeWrapper(gym.vector.VectorWrapper):
    def __init__(
        self,
        envs: gym.vector.VectorEnv,
        replay_buffer: torchrl.data.TensorDictReplayBuffer,
        max_return_to_go: float | None = None,
    ):
        assert isinstance(replay_buffer[0], Episode)

        super().__init__(envs)

        self.replay_buffer = replay_buffer

        # Constant factor of 2 was used in Online Decision Transformer paper
        max_return_to_go: float = max_return_to_go or 2 * max(
            replay_buffer, key=lambda x: x[0]
        )
        self.max_return_to_go = max_return_to_go

        self.episodes: list[Episode] = []
        self.num_finished_episodes = 0
        self._reset_episodes()

    def step(self, actions):
        """Returns the number of autoreset environments."""
        observations, rewards, terminated, truncated, info = super().step(actions)

        has_autoreset = np.logical_or(terminated, truncated)

        # Update episode stats
        zipped = zip(self.episodes, actions, observations, rewards, has_autoreset)
        for episode, action, observation, reward, reset in zipped:
            episode.step(action, observation, reward, has_stopped=reset)

        # Move finished episodes to the replay buffer
        finished_episodes = [
            (idx, ep) for idx, ep in enumerate(self.episodes) if ep.has_stopped
        ]
        indices, finished_episodes = zip(*finished_episodes)
        for episode in finished_episodes:
            episode.hindsight_label()

        self.replay_buffer.extend([ep.to_tensordict() for ep in finished_episodes])
        self._reset_episodes(indices)

        return observations, rewards, terminated, truncated, rewards, info
        # return np.sum(has_autoreset)

    def reset(self, *, seed=None, options=None):
        self._reset_episodes()
        return super().reset(seed=seed, options=options)

    def _reset_episodes(self, indices: list[int] | None = None):
        if indices is None:
            self.num_finished_episodes = 0
            self.episodes = self.env.num_envs * [
                Episode(max_return_to_go=self.max_return_to_go)
            ]
        else:
            self.num_finished_episodes -= len(indices)
            for idx in indices:
                self.episodes[idx] = Episode(max_return_to_go=self.max_return_to_go)


class AggregateWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        initial_factory: Callable,
        aggregate: Callable,
    ):
        super().__init__(env)

        self.initial_factory = initial_factory
        self.current = self.initial_factory()
        self.aggregations = []

        self.aggregate_func = aggregate

    def step(self, action):
        state = self.env.step(action)
        self.aggregate(*state)

        return state

    def aggregate(self, observations, rewards, terminated, truncated, info):
        self.current = self.aggregate_func(
            self.current, observations, rewards, terminated, truncated, info
        )

        if terminated or truncated:
            self.aggregations.append(self.current)
            self.current = self.initial_factory()

    def reset(self, *, seed=None, options=None):
        self.aggregations.append(self.current)
        self.current = self.initial_factory()
        return super().reset(seed=seed, options=options)


class VectorAggregateWrapper(gym.vector.VectorWrapper):
    def __init__(
        self, env: gym.vector.VectorEnv, wrapper: type[AggregateWrapper], **kwargs
    ):
        super().__init__(env, **kwargs)

        self.wrappers = [wrapper(gym.Env(), **kwargs) for _ in range(self.num_envs)]

    def step(self, actions):
        states = self.env.step(actions)

        for wrapper, *state in zip(self.wrappers, *states):
            wrapper.aggregate(*state)

        return states

    def reset(self, *, seed=None, options=None):
        for wrapper in self.wrappers:
            wrapper.reset(seed=seed, options=options)

        return super().reset(seed=seed, options=options)

    @property
    def aggregations(self):
        return [agg for wrapper in self.wrappers for agg in wrapper.aggregations]

    @property
    def current(self) -> list:
        return [wrapper.current for wrapper in self.wrappers]
        