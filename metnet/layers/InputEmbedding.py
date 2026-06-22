"""Input embedding layer for MetNet-3, projecting concatenated inputs to the
internal channel representation."""

import torch
from torch import nn as nn


class InputEmbedding(nn.Module):
    """
    Project concatenated input channels to the network's internal representation size.

    From the MetNet-3 paper, Section "Inputs":
    'Inputs are embedded to the internal representation of size 512 using a linear layer.'

    Since inputs are spatial grids rather than flat vectors, this linear layer is
    implemented as a 1x1 convolution, which applies the same linear projection
    independently at every spatial location.
    """

    def __init__(self, in_channels: int, out_channels: int = 512):
        """
        Args:
            in_channels: Number of input channels after concatenating all inputs
                (e.g. 793 for the 4km high-resolution path, 17 for the 8km
                low-resolution path)
            out_channels: Internal representation size, 512 as per the MetNet-3 paper
        """
        super().__init__()
        self.embed = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Concatenated input tensor of shape [Batch, in_channels, Height, Width]

        Returns:
            Tensor of shape [Batch, out_channels, Height, Width]
        """
        return self.embed(x)
