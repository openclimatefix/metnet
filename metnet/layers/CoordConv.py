"""The coordinate convolution class."""

import torch
import torch.nn as nn


class AddCoords(nn.Module):
    """argument input tensors with spatial information."""

    def __init__(self, with_r=False):
        """
        Initialize the add coordinates class.

        Args:
            with_r: bool
        """
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Add spatial information to the input tensor.

        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat(
            [
                input_tensor,
                xx_channel.type_as(input_tensor),
                yy_channel.type_as(input_tensor),
            ],
            dim=1,
        )

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2)
                + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2)
            )
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """Coordinate convolution."""

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        """
        Initialize the coordinate convolution.

        Args:
            in_channels : int
            out_channels : int,
            with_r : boolean =False,
            **kwargs : dict[str, Unknown]
        """
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        """
        Apply a forward pass to the input.

        Args:
            x: tensor
        """
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
