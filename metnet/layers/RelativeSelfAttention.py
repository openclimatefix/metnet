"""
Relative Self Attention Implementation
"""
from typing import Tuple

import torch
import torch.nn as nn


class RelativeSelfAttention(nn.Module):
    """
    Relative Self-Attention similar to Swin V1.

    Implementation inspired from ChristophReich1996's MaxViT implementation.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 32,
        attn_grid_window_size: Tuple[int, int] = (8, 8),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_normalised_qk: bool = True,
    ) -> None:
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Number of input channels
        num_heads : int, optional
            Number of attention heads, by default 32
        attn_grid_window_size : Tuple[int, int], optional
            attention grid window size, by default (8, 8)
        attn_drop : float, optional
            attention dropout rate, by default 0.0
        proj_drop : float, optional
            post attention projection dropout rate, by default 0.0
        use_normalised_qk : bool, by default True
            Normalise queries and keys, (as in Metnet 3)
        """
        super().__init__()

        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.attn_grid_window_size: Tuple[int, int] = attn_grid_window_size
        self.scale: float = num_heads**-0.5
        self.attn_area: int = attn_grid_window_size[0] * attn_grid_window_size[1]

        self.qkv_mapping = nn.Linear(
            in_features=in_channels, out_features=3 * in_channels, bias=True
        )
        self.use_normalised_qk = use_normalised_qk
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * attn_grid_window_size[0] - 1) * (2 * attn_grid_window_size[1] - 1), num_heads
            )
        )

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer(
            "relative_position_index",
            self.get_relative_position_index(attn_grid_window_size[0], attn_grid_window_size[1]),
        )

    def get_relative_position_index(self, win_h: int, win_w: int) -> torch.Tensor:
        """
        Function to generate pair-wise relative position index for each token inside the window.

        Taken from Timms Swin V1 implementation.

        Parameters
        ----------
        win_h : int
            Window/Grid height.
        win_w : int
            Window/Grid width.

        Returns:
        -------
        torch.Tensor
            relative_coords (torch.Tensor): Pair-wise relative position indexes
            [height * width, height * width].
        """
        coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += win_h - 1
        relative_coords[:, :, 1] += win_w - 1
        relative_coords[:, :, 0] *= 2 * win_w - 1
        return relative_coords.sum(-1)

    def _get_relative_positional_bias(self) -> torch.Tensor:
        """
        Returns the relative positional bias.

        Returns:
        -------
        torch.Tensor
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            input tensor of the shape [B_, N, C].

        Returns:
        -------
        torch.Tensor
            Output tensor of the shape [B_, N, C].
        """
        # Get shape of x
        B_, N, _ = x.shape
        # Perform query key value mapping
        qkv = self.qkv_mapping(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.use_normalised_qk:
            q = torch.nn.functional.normalize(q, dim=1)  # TODO: verify dim
            k = torch.nn.functional.normalize(k, dim=1)  # TODO: verify dim

        # q = q * self.scale  # TODO: verify if this should be applied after norm
        # Compute attention maps
        attn = self.softmax(q @ k.transpose(-2, -1) + self._get_relative_positional_bias())
        # Map value with attention maps
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        # Perform final projection and dropout
        output = self.proj(output)  # TODO: Check if this is needed
        output = self.proj_drop(output)
        return output
