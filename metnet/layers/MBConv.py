"""
MBConv Implementation
"""
from typing import Type

import torch
from torch import nn

from metnet.layers.SqueezeExcitation import SqueezeExcite
from metnet.layers.StochasticDepth import StochasticDepth


class MBConv(nn.Module):
    """
    MB Conv implementation, inspired by timm's implementation.
    """

    def __init__(
        self,
        in_channels: int,
        expansion_rate: int = 4,
        downscale: bool = False,
        act_layer: Type[nn.Module] = nn.GELU,
        drop_path: float = 0.0,
        kernel_size: int = 3,
        se_bottleneck_ratio: float = 0.25,
    ):
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Input channels
        expansion_rate : int, optional
            Expansion rate for the output channels, by default 4
        downscale : bool, optional
            Flag to denote downscaling in the conv branch, by default False
        act_layer : Type[nn.Module], optional
            activation layer, by default nn.GELU
        drop_path : float, optional
            Stochastic Depth ratio, by default 0.0
        kernel_size : int, optional
            Conv kernel size, by default 3
        se_bottleneck_ratio : float, optional
            Squeeze Excite reduction ratio, by default 0.25
        """
        # TODO: Verify implemtnetation
        super().__init__()
        self.in_channels = in_channels
        self.drop_path_rate = drop_path
        self.expansion_rate = expansion_rate
        self.downscale = downscale
        out_channels = self.in_channels * self.expansion_rate

        self.conv_se_branch = nn.Sequential(
            nn.LayerNorm(in_channels),  # Pre Norm
            nn.Conv2d(  # Conv 1x1
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.LayerNorm(out_channels),  # Norm1
            nn.Conv2d(  # Depth wise Conv kxk
                out_channels,
                out_channels,
                kernel_size,
                stride=2 if self.downscale else 1,
                groups=out_channels,
            ),
            nn.LayerNorm(out_channels),  # Norm2
            SqueezeExcite(
                in_channels=out_channels, act_layer=act_layer, rd_ratio=se_bottleneck_ratio
            ),
            nn.Conv2d(  # Conv 1x1
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            # No Norm as this is the last convolution layer in this block
        )

        self.stochastic_depth = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

        self.skip_path = nn.Identity()
        if self.downscale:
            self.skip_path = nn.Sequential(
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward step

        Parameters
        ----------
        X : torch.Tensor
            Input Tensor

        Returns:
        -------
        torch.Tensor
            MBConv output
        """
        conv_se_output = self.conv_se_branch(X)
        conv_se_output = self.stochastic_depth(conv_se_output)
        return conv_se_output + self.skip_path(X)
