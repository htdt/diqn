from typing import List
from collections import namedtuple
import torch
from torch.optim import Adam, Optimizer
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ExponentialLR

LRParam = namedtuple('LRParam', ('start', 'decay_rate',
                                 'update_every', 'last_update'))


class ParamOptim:
    def __init__(
            self,
            params: List[torch.Tensor],
            lr: LRParam,
            eps: float = .0003,
            clip_grad: float = None,
            optimizer: Optimizer = Adam,
            retain_graph=False,
    ):
        self.params = params
        self.clip_grad = clip_grad
        self.optim = optimizer(self.params, lr=lr.start, eps=eps)
        self.retain_graph = retain_graph
        self.lr_scheduler = ExponentialLR(self.optim, lr.decay_rate)
        self.lr = lr
        self.lr_need_update = True

    def step_lr(self, n_iter):
        if self.lr_need_update or\
            (n_iter % self.lr.update_every == 0 and
                n_iter // self.lr.update_every <= self.lr.last_update):
            self.lr_need_update = False
            ep = min(n_iter // self.lr.update_every, self.lr.last_update)
            self.lr_scheduler.step(ep)
            return self.lr_scheduler.get_lr()[0]
        else:
            return None

    def step(self, loss):
        self.optim.zero_grad()
        loss.backward(retain_graph=self.retain_graph)
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad)
        self.optim.step()
        return loss


class ParamOptimMSE(ParamOptim):
    def __init__(self, *args, **kwargs):
        super(ParamOptimMSE, self).__init__(*args, **kwargs)
        self.criterion = MSELoss()

    def step(self, x, y):
        return super().step(self.criterion(x, y))
