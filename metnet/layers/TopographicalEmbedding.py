"""Topographical Embedding for MetNet3."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopographicalEmbedding(nn.Module):
    """
    Topographical Embedding class

    "we allocate a grid of embeddings with a stride of 4 km where each point is associated with 20 scalar parameters."

    """

    def __init__(
        self,
        grid_height: int,  # number of 4km grid points in y direction
        grid_width: int,  # number of 4km grid points in x direction
        embedding_dim: int = 20,
    ):
        super().__init__()
        # Learnable parameter grid — trained like NLP embeddings
        self.embedding_grid = nn.Parameter(torch.randn(1, embedding_dim, grid_height, grid_width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W] — we just need batch size and spatial dims
        B, _, H, W = x.shape

        # Expand grid to batch size
        grid = self.embedding_grid.expand(B, -1, -1, -1)

        # Bilinear interpolation to input spatial size
        # This handles the case where input spatial size differs from grid size
        embeddings = F.interpolate(
            grid,
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )

        return embeddings  # [B, 20, H, W]
