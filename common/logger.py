import numpy as np
from tensorboardX import SummaryWriter
try:
    import nvidia_smi
except ModuleNotFoundError:
    nvidia_smi = None


class Logger:
    def __init__(self, device='cpu', eps=None):
        self.log = SummaryWriter()
        if nvidia_smi and device != 'cpu':
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None
        self.eps = eps

    def stats(self, n_iter):
        if self.eps is not None:
            self.log.add_scalar('eps', self.eps(), n_iter)

        if self.handle is not None:
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
            self.log.add_scalar('nvidia/load', res.gpu, n_iter)
            res = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
            self.log.add_scalar(
                'nvidia/mem_gb', res.used / (1024 ** 3), n_iter)

    def output(self, data_dict, n_iter):
        for key, val in data_dict.items():
            if hasattr(val, 'shape') and np.prod(val.shape) > 1:
                self.log.add_histogram(key, val, n_iter)
            else:
                self.log.add_scalar(key, val, n_iter)
