import torch
import torch.nn as nn


def lerp_nn(source: nn.Module, target: nn.Module, tau: float):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1. - tau) + s.data * tau)


def get_huber_loss(bellman_errors, kappa=1):
    be_abs = bellman_errors.abs()
    huber_loss_1 = (be_abs <= kappa).float() * 0.5 * bellman_errors ** 2
    huber_loss_2 = (be_abs > kappa).float() * kappa * (be_abs - 0.5 * kappa)
    return huber_loss_1 + huber_loss_2


def flat_grads(params):
    return torch.cat(
        [p.grad.data.flatten() for p in params if p.grad is not None])
