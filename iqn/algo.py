from dataclasses import dataclass
import copy
import torch
from common.save_load_state import SaveLoadState
from core.tools import get_huber_loss, lerp_nn, flat_grads
from core.optim import ParamOptim
from core.optim import LRParam
from iqn.model import IQN


@dataclass
class IQNAgent(SaveLoadState):
    qf: IQN
    lr: LRParam
    target_tau: float

    def __post_init__(self):
        self.init_target()
        self.optim = ParamOptim(self.qf.parameters(), self.lr)

    def init_target(self):
        self.qf_target = copy.deepcopy(self.qf)
        self.qf_target.eval()

    def _get_target_qv(self, batch):
        with torch.no_grad():
            def target_shape(x): return x.repeat(self.qf_target.num_samples, 1)
            actions = self.qf.argmax_actions(batch.next_obs)
            val = self.qf_target.samples_per_action(batch.next_obs, actions)[0]
            return target_shape(batch.cum_reward) +\
                target_shape(batch.mask_gamma) * val

    def _same_shape(self, qv, tqv, quantiles):
        num_q, num_qt = self.qf.num_samples, self.qf_target.num_samples
        return qv.view(num_q, -1).t().unsqueeze(1).repeat(1, num_qt, 1),\
            tqv.view(num_qt, -1).t().unsqueeze(2).repeat(1, 1, num_q),\
            quantiles.view(num_q, -1).t().unsqueeze(1).repeat(1, num_qt, 1)

    def _get_loss(self, q_value, target_qv, quantiles):
        bellman_errors = target_qv - q_value
        huber_loss = get_huber_loss(bellman_errors)
        neg_errors = (bellman_errors.detach() < 0).float()
        quantile_huber_loss = (quantiles - neg_errors).abs() * huber_loss
        return quantile_huber_loss.sum(2).mean()

    def update(self, batch):
        qv, quantiles = self.qf.samples_per_action(batch.obs, batch.action)
        tqv = self._get_target_qv(batch)
        qv, tqv, quantiles = self._same_shape(qv, tqv, quantiles)
        loss = self._get_loss(qv, tqv, quantiles)
        self.optim.step(loss)

        grad_qf = flat_grads(self.qf.parameters())
        lerp_nn(self.qf, self.qf_target, self.target_tau)
        return {
            'loss': loss,
            'q_value/mean': qv.mean(),
            'q_value/std': qv.std(),
            'grad/std': grad_qf.std(),
            'grad/max': grad_qf.max(),
        }

    def _list_stateful(self):
        return [self.qf]
