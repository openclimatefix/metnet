"""
Implementation of Partition (Grid and Block) Attention
"""
from typing import Tuple, Type

import torch
from torch import Tensor, nn

from metnet.layers.RelativeSelfAttention import RelativeSelfAttention
from metnet.layers.StochasticDepth import StochasticDepth


class PartitionAttention(nn.Module):
    """
    Partition Attention

    Implements the common functionality for block and grid attention.
    X ← X + StochasticDepth(RelAttention(PreNorm(Partition(X))))
    X ← X + StochasticDepth(MLP(PostNorm(X)))
    X ← ReversePartition(X)

    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 32,
        attn_grid_window_size: Tuple[int, int] = (8, 8),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        pre_norm_layer: Type[nn.Module] = nn.LayerNorm,
        post_norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp: Type[nn.Module] = None,
        use_normalised_qk: bool = True,
    ) -> None:
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        num_heads : int, optional
             Number of attention heads, by default 32
        attn_grid_window_size : Tuple[int, int], optional
             Grid/Window size to be utilized, by default (8, 8)
        attn_drop : float, optional
             Dropout ratio of attention weight, by default 0.0
        proj_drop : float, optional
            Dropout ratio of output, by default 0.0
        drop_path : float, optional
            Stochastic depth, by default 0.0
        pre_norm_layer : Type[nn.Module], optional
            Pre norm layer, by default nn.LayerNorm
        post_norm_layer : Type[nn.Module], optional
            Post norm layer, by default nn.LayerNorm
        mlp : Type[nn.Module], optional
            MLP to be used after the attention, by default None
        use_normalised_qk : bool, optional
            Normalise queries and keys as done in Metnet 3, by default True.

        Notes:
        -----
        Specific to Metnet 3 implementation
        TODO: Add the MLP as an optional parameter.
        """
        super().__init__()
        # Save parameters
        self.attn_grid_window_size: Tuple[int, int] = attn_grid_window_size
        # Init layers
        self.attention = RelativeSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            attn_grid_window_size=attn_grid_window_size,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_normalised_qk=use_normalised_qk,
        )
        self.pre_norm_layer = pre_norm_layer(in_channels)
        self.post_norm_layer = post_norm_layer(in_channels)

        if mlp:
            # TODO: allow for an mlp to be passed here
            raise NotImplementedError("Metnet 3 does noes use MLPs in MaxVit.")
        else:
            self.mlp = nn.Identity()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def partition_function(self, X: torch.Tensor):
        """
        Partition function.

        To be overridden by block or grid partition

        Parameters
        ----------
        X : torch.Tensor
            Input tensor

        Raises:
        ------
        NotImplementedError
            Should not be called without implementation in the child class
        """
        raise NotImplementedError

    def reverse_function(
        self,
        partitioned_input: torch.Tensor,
        original_size: Tuple[int, int],
    ):
        """
        Undo Partition

        To be overridden by functions reversing the block or grid partitions

        Parameters
        ----------
        partitioned_input : torch.Tensor
            Partitioned input
        original_size : Tuple[int, int]
            Original Input size

        Raises:
        ------
        NotImplementedError
            Should not be called without implementation in the child class
        """
        raise NotImplementedError

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of the shape [N, C_in, H, W].

        Returns:
        -------
        torch.Tensor
            Output tensor of the shape [N, C_out, H (// 2), W (// 2)].
        """
        # Save original shape
        _, C, H, W = X.shape
        # Perform partition
        input_partitioned = self.partition_function(X)
        input_partitioned = input_partitioned.view(
            -1, self.attn_grid_window_size[0] * self.attn_grid_window_size[1], C
        )
        # Perform normalization, attention, and dropout
        output = input_partitioned + self.drop_path(
            self.attention(self.pre_norm_layer(input_partitioned))
        )

        # Perform normalization, MLP, and dropout
        output = output + self.drop_path(self.mlp(self.post_norm_layer(output)))

        # Reverse partition
        output = self.reverse_function(output, (H, W))
        return output


class BlockAttention(PartitionAttention):
    """
    Block Attention.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 32,
        attn_grid_window_size: Tuple[int, int] = (8, 8),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        pre_norm_layer: Type[nn.Module] = nn.LayerNorm,
        post_norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp: Type[nn.Module] = None,
        use_normalised_qk: bool = True,
    ) -> None:
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        num_heads : int, optional
             Number of attention heads, by default 32
        attn_grid_window_size : Tuple[int, int], optional
             Grid/Window size for attention, by default (8, 8)
        attn_drop : float, optional
             Dropout ratio of attention weight, by default 0.0
        proj_drop : float, optional
            Dropout ratio of output, by default 0.0
        drop_path : float, optional
            Stochastic depth, by default 0.0
        pre_norm_layer : Type[nn.Module], optional
            Pre norm layer, by default nn.LayerNorm
        post_norm_layer : Type[nn.Module], optional
            Post norm layer, by default nn.LayerNorm
        mlp : Type[nn.Module], optional
            MLP to be used after the attention, by default None
        use_normalised_qk : bool, optional
            Normalise queries and keys as done in Metnet 3, by default True.
        """
        super().__init__(
            in_channels,
            num_heads,
            attn_grid_window_size,
            attn_drop,
            proj_drop,
            drop_path,
            pre_norm_layer,
            post_norm_layer,
            mlp,
            use_normalised_qk,
        )

    def partition_function(self, input: Tensor) -> torch.Tensor:
        """
        Block partition function.

        Parameters
        ----------
        input : Tensor
            input (torch.Tensor): Input tensor of the shape [N, C, H, W].

        Returns:
        -------
        torch.Tensor
            blocks (torch.Tensor): Unfolded input tensor of the shape
            [N * blocks, partition_size[0], partition_size[1], C].
        """
        # Get size of input
        N, C, H, W = input.shape
        # Unfold input
        blocks = input.view(
            N,
            C,
            H // self.attn_grid_window_size[0],
            self.attn_grid_window_size[0],
            W // self.attn_grid_window_size[1],
            self.attn_grid_window_size[1],
        )
        # Permute and reshape to
        # [N * blocks, self.attn_grid_window_size[0], self.attn_grid_window_size[1], channels]
        blocks = (
            blocks.permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(-1, self.attn_grid_window_size[0], self.attn_grid_window_size[1], C)
        )
        return blocks

    def reverse_function(
        self,
        partitioned_input: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Reverses the block partition.

        Parameters
        ----------
        partitioned_input : torch.Tensor
            Block tensor of the shape
            [N * partitioned_input, partition_size[0], partition_size[1], C].
        original_size : Tuple[int, int]
            Original shape.

        Returns:
        -------
        torch.Tensor
            output (torch.Tensor): Folded output tensor of the shape
            [N, C, original_size[0], original_size[1]].
        """
        # Get height and width
        H, W = original_size
        # Compute original batch size
        N = int(
            partitioned_input.shape[0]
            / (H * W / self.attn_grid_window_size[0] / self.attn_grid_window_size[1])
        )
        # Fold grid tensor
        output = partitioned_input.view(
            N,
            H // self.attn_grid_window_size[0],
            W // self.attn_grid_window_size[1],
            self.attn_grid_window_size[0],
            self.attn_grid_window_size[1],
            -1,
        )
        output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(N, -1, H, W)
        return output


class GridAttention(PartitionAttention):
    """
    Grid Attention
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 32,
        attn_grid_window_size: Tuple[int, int] = (8, 8),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        pre_norm_layer: Type[nn.Module] = nn.LayerNorm,
        post_norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp: Type[nn.Module] = None,
        use_normalised_qk: bool = True,
    ) -> None:
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        num_heads : int, optional
             Number of attention heads, by default 32
        attn_grid_window_size : Tuple[int, int], optional
             Grid/Window size to be utilized, by default (8, 8)
        attn_drop : float, optional
             Dropout ratio of attention weight, by default 0.0
        proj_drop : float, optional
            Dropout ratio of output, by default 0.0
        drop_path : float, optional
            Stochastic depth, by default 0.0
        pre_norm_layer : Type[nn.Module], optional
            Pre norm layer, by default nn.LayerNorm
        post_norm_layer : Type[nn.Module], optional
            Post norm layer, by default nn.LayerNorm
        mlp : Type[nn.Module], optional
            MLP to be used after the attention, by default None
        use_normalised_qk : bool, optional
            Normalise queries and keys as done in Metnet 3, by default True.
        """
        super().__init__(
            in_channels,
            num_heads,
            attn_grid_window_size,
            attn_drop,
            proj_drop,
            drop_path,
            pre_norm_layer,
            post_norm_layer,
            mlp,
            use_normalised_qk,
        )

    def partition_function(self, input: Tensor) -> torch.Tensor:
        """
        Grid partition function.

        Parameters
        ----------
        input : Tensor
            Input tensor of the shape [N, C, H, W].

        Returns:
        -------
        torch.Tensor
            Unfolded input tensor of the shape
            [N * grids, grid_size[0], grid_size[1], C].
        """
        # Get size of input
        N, C, H, W = input.shape
        # Unfold input
        grid = input.view(
            N,
            C,
            self.attn_grid_window_size[0],
            H // self.attn_grid_window_size[0],
            self.attn_grid_window_size[1],
            W // self.attn_grid_window_size[1],
        )
        # Permute and reshape [N * (H // self.attn_grid_window_size[0]) * (W // self.attn_grid_window_size[1]), self.attn_grid_window_size[0], window_size[1], C]  # noqa
        grid = (
            grid.permute(0, 3, 5, 2, 4, 1)
            .contiguous()
            .view(-1, self.attn_grid_window_size[0], self.attn_grid_window_size[1], C)
        )
        return grid

    def reverse_function(
        self,
        partitioned_input: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Reverses the grid partition.

        Parameters
        ----------
        partitioned_input : torch.Tensor
            Grid tensor of the shape
            [N * partitioned_input, partition_size[0], partition_size[1], C].
        original_size : Tuple[int, int]
            Original shape.

        Returns:
        -------
        torch.Tensor
            Folded output tensor of the shape [N, C, original_size[0], original_size[1]].
        """
        # Get height, width, and channels
        (H, W), C = original_size, partitioned_input.shape[-1]
        # Compute original batch size
        N = int(
            partitioned_input.shape[0]
            / (H * W / self.attn_grid_window_size[0] / self.attn_grid_window_size[1])
        )
        # Fold partitioned_input tensor
        output = partitioned_input.view(
            N,
            H // self.attn_grid_window_size[0],
            W // self.attn_grid_window_size[1],
            self.partition_window_size[0],
            self.partition_window_size[1],
            C,
        )
        output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(N, C, H, W)
        return output
