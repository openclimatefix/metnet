import torch
import torch.nn as nn
from .layers.utils import get_conv_layer
from .layers import ConvGRU, TimeDistributed, ConditionTime, MetNetPreprocessor
from axial_attention import AxialAttention
import antialiased_cnns


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


class MetNet(torch.nn.Module):
    def __init__(
            self,
            image_encoder: str = "downsampler",
            input_channels: int = 12,
            sat_channels: int = 12,
            input_size: int = 256,
            output_channels: int = 12,
            hidden_dim: int = 64,
            kernel_size: int = 3,
            num_layers: int = 1,
            num_att_layers: int = 1,
            head: nn.Module = nn.Identity(),
            forecast_steps: int = 48,
            temporal_dropout: float = 0.2,
    ):
        super(MetNet, self).__init__()
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.preprocessor = MetNetPreprocessor(
            sat_channels=sat_channels, crop_size=input_size, use_space2depth=True, split_input=True
        )
        # Update number of input_channels with output from MetNetPreprocessor
        new_channels = sat_channels * 4  # Space2Depth
        new_channels *= 2  # Concatenate two of them together
        input_channels = input_channels - sat_channels + new_channels
        self.drop = nn.Dropout(temporal_dropout)
        if image_encoder in ["downsampler", "default"]:
            image_encoder = DownSampler(input_channels + forecast_steps)
        else:
            raise ValueError(f"Image_encoder {image_encoder} is not recognized")
        self.image_encoder = TimeDistributed(image_encoder)
        self.ct = ConditionTime(forecast_steps)
        self.temporal_enc = TemporalEncoder(
            image_encoder.output_channels, hidden_dim, ks=kernel_size, n_layers=num_layers
        )
        self.temporal_agg = nn.Sequential(
            *[
                AxialAttention(dim=hidden_dim, dim_index=1, heads=8, num_dimensions=2)
                for _ in range(num_att_layers)
            ]
        )

        self.head = head
        self.head = nn.Conv2d(hidden_dim, output_channels, kernel_size=(1, 1))  # Reduces to mask
        # self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), )

    def encode_timestep(self, x, fstep=1):

        # Preprocess Tensor
        x = self.preprocessor(x)

        # Condition Time
        x = self.ct(x, fstep)

        ##CNN
        x = self.image_encoder(x)

        # Temporal Encoder
        _, state = self.temporal_enc(self.drop(x))
        return self.temporal_agg(state)

    def forward(self, imgs):
        """It takes a rank 5 tensor
        - imgs [bs, seq_len, channels, h, w]
        """

        # Compute all timesteps, probably can be parallelized
        res = []
        for i in range(self.forecast_steps):
            x_i = self.encode_timestep(imgs, i)
            out = self.head(x_i)
            res.append(out)
        res = torch.stack(res, dim=1)
        return res


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=384, ks=3, n_layers=1):
        super().__init__()
        self.rnn = ConvGRU(in_channels, out_channels, (ks, ks), n_layers, batch_first=True)

    def forward(self, x):
        x, h = self.rnn(x)
        return (x, h[-1])


def feat2image(x, target_size=(128, 128)):
    "This idea comes from MetNet"
    x = x.transpose(1, 2)
    return x.unsqueeze(-1).unsqueeze(-1) * x.new_ones(1, 1, 1, *target_size)
