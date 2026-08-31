"""TorchSparse layers retaining the SpConv checkpoint parameter layout."""

import math

import torch
from torch import nn
from torchsparse import SparseTensor
from torchsparse.nn import functional as spf
from torchsparse.utils import make_ntuple


class SparseConv3d(nn.Module):
    """Stride-one TorchSparse convolution compatible with ``spconv.SubMConv3d``.

    Parameters intentionally use SpConv's ``[kz, ky, kx, in, out]`` layout.
    This preserves model keys, existing checkpoint tensors, and pruning masks;
    TorchSparse receives an inexpensive flattened view during the forward pass.
    """

    def __init__(self, in_channels, out_channels, kernel_size, indice_key=None,
                 bias=False, dilation=1, config=None):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = (1, 1, 1)
        self.dilation = make_ntuple(dilation, ndim=3)
        self.padding = tuple((size - 1) // 2 for size in self.kernel_size)
        self.indice_key, self.config = indice_key, config
        bound = 1.0 / math.sqrt(in_channels * math.prod(self.kernel_size))
        self.weight = nn.Parameter(torch.empty(
            *self.kernel_size, in_channels, out_channels))
        nn.init.uniform_(self.weight, -bound, bound)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: SparseTensor) -> SparseTensor:
        if self.kernel_size == (1, 1, 1):
            weight = self.weight.reshape(self.in_channels, self.out_channels)
        else:
            weight = self.weight.reshape(-1, self.in_channels, self.out_channels)
        return spf.conv3d(input, weight=weight, kernel_size=self.kernel_size,
                          bias=self.bias, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, config=self.config,
                          training=self.training)


def sparse_tensor(features, coordinates, spatial_shape, batch_size):
    """Build a TorchSparse tensor from this project's B-Z-Y-X coordinates."""
    return SparseTensor(features, coordinates.int(), spatial_range=(
        int(batch_size), *tuple(int(value) for value in spatial_shape)))
