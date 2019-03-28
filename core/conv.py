from typing import List, Tuple
import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(
            self,
            input_size: Tuple[int],
            channels: List[int],
            kernel_size: List[int],
            stride: List[int],
    ):
        super(Conv, self).__init__()
        assert len(channels) == len(kernel_size) == len(stride)
        self.conv = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(c_in, c_out, ker, st), nn.ReLU())
            for c_in, c_out, ker, st in zip([input_size[0]] + channels[:-1],
                                            channels, kernel_size, stride)])
        with torch.no_grad():
            tmp = torch.zeros((1,) + input_size)
            self.output_size = len(self.conv(tmp).view(-1))

    def forward(self, x):
        return self.conv(x).view(x.shape[0], -1)
