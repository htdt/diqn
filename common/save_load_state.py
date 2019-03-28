import abc
import torch


class SaveLoadState(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _list_stateful(self):
        raise NotImplementedError

    def save(self, path):
        dicts = list(map(lambda x: x.state_dict(), self._list_stateful()))
        torch.save(dicts, path)

    def load(self, path, map_location=None):
        params = torch.load(path, map_location=map_location)
        for nn, param in zip(self._list_stateful(), params):
            nn.load_state_dict(param)
