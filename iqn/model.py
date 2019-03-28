from typing import List
import math
import torch
import torch.nn as nn
from core.conv import Conv


class IQN(nn.Module):
    def __init__(
            self,
            hidden_sizes: List[int],
            input_size: int,
            output_size: int,
            quantile_dim: int,
            num_samples: int,
            num_samples_eval: int,
            device: str,
            conv: Conv = None,
    ):
        super(IQN, self).__init__()

        self.quantile_dim = quantile_dim
        self.num_samples = num_samples
        self.num_samples_eval = num_samples_eval
        self.device = device

        self.conv = conv
        if conv is not None:
            input_size = conv.output_size
        self.quantile_fc = nn.Sequential(
            nn.Linear(quantile_dim, input_size), nn.ReLU())
        self.linear = nn.Sequential(*[
            nn.Sequential(nn.Linear(s_in, s_out), nn.ReLU())
            for s_in, s_out in zip(
                [input_size] + hidden_sizes[:-1], hidden_sizes)])

        self.adv = nn.Linear(hidden_sizes[-1], output_size)
        self.val = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x, num_quantiles):
        if self.conv:
            x = self.conv(x)
        x = x.repeat(num_quantiles, 1)
        quantile, uniform = self._get_quantile(x.shape[0])
        x *= self.quantile_fc(quantile)
        x = self.linear(x)
        adv, val = self.adv(x), self.val(x)
        q = val + adv - adv.mean(1, keepdim=True)
        return q, uniform

    def _get_quantile(self, size):
        tparam = {'dtype': torch.float32, 'device': self.device}
        uniform = torch.empty(size, 1, **tparam).uniform_()
        uniform_repeated = uniform.repeat(1, self.quantile_dim)
        embedding_range = torch.arange(1, self.quantile_dim + 1, **tparam)\
            .unsqueeze(0)
        return torch.cos(embedding_range * math.pi * uniform_repeated), uniform

    def argmax_actions(self, x):
        return self(x, self.num_samples_eval)[0]\
            .view(self.num_samples_eval, x.shape[0], -1).mean(0).argmax(1)

    def samples_per_action(self, x, action):
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        action = action.repeat(self.num_samples, 1)
        q, uniform = self(x, self.num_samples)
        return q.gather(1, action), uniform
