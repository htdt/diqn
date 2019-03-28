from typing import Tuple
from dataclasses import dataclass
from collections import namedtuple
import random
import torch

Batch = namedtuple('Batch', ('obs', 'next_obs',
                             'action', 'cum_reward', 'mask_gamma'))


@dataclass
class ReplayBuffer:
    size: int
    batch_size: int
    obs_shape: Tuple[int]
    device: str
    obs_dtype: torch.tensor
    num_env: int
    gamma: float
    n_step: int
    action_shape: Tuple[int] = (1,)

    def __post_init__(self):
        self.maxlen = self.size // self.num_env
        self._idx, self._len = 0, 0

        mem_dtypes = {
            'obs': self.obs_dtype,
            'action': torch.uint8,
            'reward': torch.float,
            'terminal': torch.uint8,
        }
        shapes = {
            'obs': (self.maxlen, self.num_env, *self.obs_shape),
            'action': (self.maxlen, self.num_env, *self.action_shape),
            'reward': (self.maxlen, self.num_env, 1),
            'terminal': (self.maxlen, self.num_env, 1),
        }
        self._buffer = {
            field: torch.empty(
                shapes[field], dtype=mem_dtypes[field],
                device=self.device)
            for field in mem_dtypes
        }
        self.gamma_space = torch.tensor(
                [self.gamma ** i for i in range(self.n_step)]
            ).unsqueeze(0).repeat(self.batch_size, 1).to(device=self.device)

    def insert(self, idx, env_step):
        for name, val in env_step.items():
            if val is not None:
                self._buffer[name][idx] = val

    def append(self, env_step):
        if 'action' in env_step:  # otherwise it's first reset() step
            self.insert(self._idx, {k: env_step[k] for k in [
                        'action', 'reward', 'terminal']})
            self._idx = (self._idx + 1) % self.maxlen
            self._len = min(self._len + 1, self.maxlen)

        # next obs, on next idx
        self.insert(self._idx, {'obs': env_step['obs']})

    def _get_masks(self, selector_n):
        term = self._buffer['terminal'][selector_n]\
            .view(self.batch_size, self.n_step).float()
        mask_reward = torch.ones_like(term)
        mask_reward[:, 1:] -= term[:, :-1]
        for i in range(1, self.n_step - 1):
            mask_reward[:, i + 1] *= mask_reward[:, i]
        mask_val = (term.sum(1, keepdim=True) == 0).float()
        return mask_val, mask_reward

    def _get_cum_reward(self, selector_n, mask):
        reward = self._buffer['reward'][selector_n].view(
            self.batch_size, self.n_step).clone()
        reward *= self.gamma_space * mask
        return reward.sum(1, keepdim=True)

    def sample(self):
        selector = self._sample_selector()
        selector_n_step = self._selector_n_step(*selector)
        mask_val, mask_reward = self._get_masks(selector_n_step)
        cum_reward = self._get_cum_reward(selector_n_step, mask_reward)
        selector_next = [(idx + self.n_step) % self.maxlen
                         for idx in selector[0]], selector[1]
        return Batch(
            obs=self._buffer['obs'][selector].float(),
            next_obs=self._buffer['obs'][selector_next].float(),
            action=self._buffer['action'][selector].long(),
            cum_reward=cum_reward,
            mask_gamma=mask_val * (self.gamma ** self.n_step),
        )

    def _selector_n_step(self, idx0, idx1):
        idx0 = [(i + n) % self.maxlen
                for i in idx0 for n in range(self.n_step)]
        idx1 = [i for i in idx1 for _ in range(self.n_step)]
        return idx0, idx1

    def _sample_selector(self):
        no_next = {(self._idx - i - 1) % self.maxlen
                   for i in range(self.n_step - 1)}
        all_idx = list(set(range(len(self))) - no_next)
        idx0 = random.choices(all_idx, k=self.batch_size)
        idx1 = random.choices(range(self.num_env), k=self.batch_size)
        return idx0, idx1

    def __len__(self):
        return self._len
