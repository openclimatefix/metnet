"""The lead time conditioner module."""

import torch
import torch.nn as nn


class LeadTimeConditioner(nn.Module):
    """
    Lead time conditioner class.

    The lead time conditioner for MetNet-2, based on 'FiLM: Visual
    Reasoning with a General Conditioning Layer.

    Paper: https://arxiv.org/pdf/1709.07871.pdf

    """

    def forward(self, x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Add the conditioning to the input tensor.

        Args:
            x: Input tensor
            scale: Scale parameter
            bias: Bias parameter

        Returns:
            Input tensor with the scale multiplied to it and bias added
        """
        scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)
        bias = bias.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (scale * x) + bias
