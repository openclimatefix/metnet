"""
Relative Self Attention Implementation
"""
from typing import Tuple, Type

import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    def __init__(self, attn_size: Tuple[int, int], num_heads: int) -> None:
        super().__init__()
        self.attn_size = attn_size
        self.num_heads = num_heads

        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.attn_size[0] - 1) * (2 * self.attn_size[1] - 1), self.num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer(
            "relative_position_index",
            self.get_relative_position_index(self.attn_size[0], self.attn_size[1]),
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

    def _get_relative_positional_bias(self, attn_area: int) -> torch.Tensor:
        """
        Returns the relative positional bias.

        Returns:
        -------
        torch.Tensor
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(attn_area, attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, attn_area: int) -> torch.Tensor:
        return self._get_relative_positional_bias(attn_area=attn_area)


class RelativeSelfAttention(nn.Module):
    """
    Relative Self-Attention similar to Swin V1.

    Implementation inspired from timm's MaxViT implementation.
    """

    def __init__(
        self,
        in_channels: int,
        attention_head_dim: int = 512,
        num_heads: int = 32,
        head_first: bool = False,
        attn_grid_window_size: Tuple[int, int] = (8, 8),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_normalised_qk: bool = True,
        rel_attn_bias: Type[nn.Module] = None,
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
        self.attention_head_dim = attention_head_dim
        self.head_first = head_first
        self.attn_grid_window_size: Tuple[int, int] = attn_grid_window_size
        self.scale: float = num_heads**-0.5
        self.attn_area: int = attn_grid_window_size[0] * attn_grid_window_size[1]
        self.rel_attn_bias = rel_attn_bias

        self.qkv = nn.Conv2d(self.in_channels, self.attention_head_dim * 3, 1)

        self.use_normalised_qk = use_normalised_qk
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Conv2d(self.attention_head_dim, self.in_channels, 1)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            input tensor of the shape [N, C, H, W].

        Returns:
        -------
        torch.Tensor
            Output tensor of the shape [N, C, H, W].
        """
        # Get shape of X
        B, C, H, W = X.shape

        if self.head_first:
            q, k, v = (
                self.qkv(X).view(B, self.num_heads, self.attention_head_dim * 3, -1).chunk(3, dim=2)
            )
        else:
            q, k, v = (
                self.qkv(X).reshape(B, 3, self.num_heads, self.attention_head_dim, -1).unbind(1)
            )

        if self.use_normalised_qk:
            q = torch.nn.functional.normalize(q, dim=1)  # TODO: verify dim
            k = torch.nn.functional.normalize(k, dim=1)  # TODO: verify dim

        # q = q * self.scale  # TODO: verify if this should be applied after norm

        # Compute attention maps
        attn = q.transpose(-2, -1) @ k
        if self.rel_attn_bias is not None:
            attn = attn + self.rel_attn_bias()

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # Map value with attention maps
        output = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        # Perform final projection and dropout
        output = self.proj(output)  # TODO: Check if this is needed
        output = self.proj_drop(output)
        return output
