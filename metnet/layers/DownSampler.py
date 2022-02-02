import antialiased_cnns
import torch.nn as nn

from metnet.layers.utils import get_conv_layer


class DownSampler(nn.Module):
    def __init__(self, in_channels, output_channels: int = 256, conv_type: str = "standard"):
        super().__init__()
        conv2d = get_conv_layer(conv_type=conv_type)
        self.output_channels = output_channels
        if conv_type == "antialiased":
            antialiased = True
        else:
            antialiased = False

        self.module = nn.Sequential(
            conv2d(in_channels, 160, 3, padding=1),
            nn.MaxPool2d((2, 2), stride=1 if antialiased else 2),
            antialiased_cnns.BlurPool(160, stride=2) if antialiased else nn.Identity(),
            nn.BatchNorm2d(160),
            conv2d(160, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            conv2d(output_channels, output_channels, 3, padding=1),
            nn.MaxPool2d((2, 2), stride=1 if antialiased else 2),
            antialiased_cnns.BlurPool(output_channels, stride=2) if antialiased else nn.Identity(),
        )

    def forward(self, x):
        return self.module.forward(x)
