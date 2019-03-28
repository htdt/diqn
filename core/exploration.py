from dataclasses import dataclass
import numpy as np


@dataclass
class DecayingEpsilon:
    epsilon: float
    warmup: int
    decay_period: float
    n_iter: int = 0

    def update(self, n_iter):
        self.n_iter = n_iter

    def __call__(self):
        steps_left = self.decay_period + self.warmup - self.n_iter
        bonus = (1.0 - self.epsilon) * steps_left / self.decay_period
        bonus = np.clip(bonus, 0., 1. - self.epsilon)
        return self.epsilon + bonus
