"""Condition with time how MetNet-3 does it, with FiLM layers."""

import einops
import torch
from torch import nn as nn


class ConditionWithTimeMetNet3(nn.Module):
    """Compute Scale and bias for conditioning on time."""

    def __init__(
        self, forecast_steps: int = 722, hidden_dim: int = 32, num_feature_maps: int = 512
    ):
        """
        Compute the scale and bias factors for conditioning convolutional blocks on forecast time.

        Args:
            forecast_steps: Number of forecast steps
            hidden_dim: Hidden dimension size
            num_feature_maps: Max number of channels in thec blocks, to generate enough
            scale+bias values
                This means extra values will be generated, but keeps implementation simpler
        """
        super().__init__()
        self.forecast_steps = forecast_steps
        self.num_feature_maps = num_feature_maps
        self.relu = nn.ReLU()
        self.lead_time_network = nn.ModuleList(
            [
                nn.Linear(in_features=forecast_steps, out_features=hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=2 * num_feature_maps),
            ]
        )
        # Initialize second layer as identity function
        nn.init.zeros_(self.lead_time_network[1].weight)
        nn.init.zeros_(self.lead_time_network[1].bias)
        self.lead_time_network[1].bias.data[0::2] = 1.0

    def forward(self, x: torch.Tensor, timestep: int) -> [torch.Tensor, torch.Tensor]:
        """
        Get the scale and bias for the conditioning layers.

        From the FiLM paper, each feature map (i.e. channel) has its own scale and bias layer,
        so needs a scale and bias for each feature map to be generated

        Args:
            x: The Tensor that is used
            timestep: Index of the timestep to use, between 0 and forecast_steps

        Returns:
            2 Tensors of shape (Batch, num_feature_maps)
        """
        # One hot encode the timestep
        timesteps = torch.zeros(x.size()[0], self.forecast_steps, dtype=torch.long).type_as(x)
        timesteps[:, timestep] = 1
        # Get scales and biases
        timesteps = self.lead_time_network[0](timesteps)
        timesteps = self.relu(timesteps)
        timesteps = self.lead_time_network[1](timesteps)
        scales_and_biases = timesteps
        scales_and_biases = einops.rearrange(
            scales_and_biases,
            "b (block sb) -> b block sb",
            block=self.num_feature_maps,
            sb=2,
        )
        return scales_and_biases[:, :, 0], scales_and_biases[:, :, 1]
