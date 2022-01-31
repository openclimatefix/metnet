"""Dilated Time Conditioned Residual Convolution Block for MetNet-2"""
import torch
import torch.nn as nn


class DilatedResidualConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int = 384,
        dilation: int = 1,
        kernel_size: int = 3,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.dilated_conv_one = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            dilation=dilation,
            kernel_size=kernel_size,
        )
        # TODO Pass in the non-batch size things for layer norm
        self.layer_norm_one = nn.LayerNorm(normalized_shape=(output_channels,))
        # Target Time index conditioning

        self.activation = activation
        # TODO Check if same number of input and output channels
        self.dilated_conv_two = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            dilation=dilation,
            kernel_size=kernel_size,
        )
        self.layer_norm_two = nn.LayerNorm(normalized_shape=(output_channels,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dilated_conv_one(x)
        out = self.layer_norm_one(out)
        # TODO Add target time conditioning
        out = self.activation(out)
        out = self.dilated_conv_two(out)
        out = self.layer_norm_two(out)
        # TODO Add target time conditioning
        out = self.activation(out)
        return x + out


class UpsampleResidualConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int = 512,
        dilation: int = 1,
        kernel_size: int = 3,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.dilated_conv_one = nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=output_channels,
            stride=2,
            kernel_size=kernel_size,
        )
        # TODO Pass in the non-batch size things for layer norm
        self.layer_norm_one = nn.LayerNorm(normalized_shape=(output_channels,))
        # Target Time index conditioning

        self.activation = activation
        # TODO Check if same number of input and output channels
        self.dilated_conv_two = nn.ConvTranspose2d(
            in_channels=output_channels,
            out_channels=output_channels,
            stride=2,
            kernel_size=kernel_size,
        )
        self.layer_norm_two = nn.LayerNorm(normalized_shape=(output_channels,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dilated_conv_one(x)
        out = self.layer_norm_one(out)
        # TODO Add target time conditioning
        out = self.activation(out)
        out = self.dilated_conv_two(out)
        out = self.layer_norm_two(out)
        # TODO Add target time conditioning
        out = self.activation(out)
        return x + out
