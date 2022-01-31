import torch
import torch.nn as nn
import torchvision.transforms
import einops
from typing import List

from metnet.layers import (
    ConditionTime,
    ConvGRU,
    DownSampler,
    MetNetPreprocessor,
    TimeDistributed,
    DilatedResidualConv,
    UpsampleResidualConv,
    ConvLSTM,
)


class MetNet2(torch.nn.Module):
    def __init__(
        self,
        image_encoder: str = "downsampler",
        input_channels: int = 12,
        lstm_channels: int = 128,
        encoder_channels: int = 384,
        upsampler_channels: int = 512,
        lead_time_features: int = 2048,
        lead_time_layers: int = 2,
        num_upsampler_blocks: int = 2,
        num_context_blocks: int = 3,
        encoder_dilations: List[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        sat_channels: int = 12,
        input_size: int = 256,
        output_channels: int = 12,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 1,
        center_crop_size: int = 128,
        forecast_steps: int = 48,
        temporal_dropout: float = 0.2,
    ):
        """
        MetNet-2 builds on MetNet-1 to use an even larger context area to predict up to 12 hours ahead.

        Paper: https://arxiv.org/pdf/2111.07470.pdf

        The architecture of MetNet-2 differs from the original MetNet in terms of the axial attention is dropped, and there
        is more dilated convolutions instead.



        :param image_encoder:
        :param input_channels:
        :param sat_channels:
        :param input_size:
        :param output_channels:
        :param hidden_dim:
        :param kernel_size:
        :param num_layers:
        :param head:
        :param forecast_steps:
        :param temporal_dropout:
        """
        super(MetNet2, self).__init__()
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

        # ConvLSTM with 13 timesteps, 128 LSTM channels, 18 encoder blocks, 384 encoder channels,
        self.conv_lstm = ConvLSTM(
            input_dim=input_channels, hidden_dim=lstm_channels, kernel_size=3, num_layers=13
        )

        # Lead time network layers that generate a bias and scale vector for the lead time
        self.lead_time_network = nn.ModuleList(
            [
                nn.Linear(in_features=forecast_steps, out_features=lead_time_features),
                nn.Linear(in_features=lead_time_features, out_features=2),
            ]
        )

        # Convolutional Residual Blocks going from dilation of 1 to 128 with 384 channels
        # 3 stacks of 8 blocks form context aggregating part of arch -> only two shown in image, so have both
        self.context_block_one = nn.ModuleList(
            [
                DilatedResidualConv(
                    input_channels=lstm_channels,
                    output_channels=encoder_channels,
                    kernel_size=3,
                    dilation=d,
                )
                for d in encoder_dilations
            ]
        )
        self.context_blocks = nn.ModuleList()
        for block in range(num_context_blocks):
            self.context_blocks.append(
                nn.ModuleList(
                    [
                        DilatedResidualConv(
                            input_channels=encoder_channels,
                            output_channels=encoder_channels,
                            kernel_size=3,
                            dilation=d,
                        )
                        for d in encoder_dilations
                    ]
                )
            )

        # Center crop the output
        self.center_crop = torchvision.transforms.CenterCrop(size=center_crop_size)

        # Then tile 4x4 back to original size
        # This seems like it would mean something like this, with two applications of a simple upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # The paper though, under the architecture, has 2 upsample blocks with 512 channels, indicating it might be a
        # transposed convolution instead?
        # 2 Upsample blocks with 512 channels
        self.upsample_blocks = nn.ModuleList(
            UpsampleResidualConv(
                input_channels=encoder_channels, output_channels=upsampler_channels, kernel_size=3
            )
            for _ in range(num_upsampler_blocks)
        )

        # Shallow network of Conv Residual Block Dilation 1 with the lead time MLP embedding added
        self.residual_block_three = nn.ModuleList(
            [
                DilatedResidualConv(
                    input_channels=upsampler_channels,
                    output_channels=encoder_channels,
                    kernel_size=3,
                    dilation=1,
                )
                for _ in range(8)
            ]
        )

        # Last layers are a Conv 1x1 with 4096 channels then softmax
        self.head = nn.Conv2d(hidden_dim, output_channels, kernel_size=(1, 1))

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

        # Get Center Crop
        res = self.center_crop(res)
        # Tile 4x4
        res = einops.repeat(res, "c t h w -> c t (h h2) (w w2)", h2=4, w2=4)

        # Shallow network
        res = self.residual_block_three(res)

        # Return 1x1 Conv
        res = self.head(res)

        # Softmax for rain forecasting
        return res


class TemporalEncoder(nn.Module):
    # TODO New temporal encoder for MetNet-2 with
    """
    The input to MetNet-2 captures 2048 km×2048 km of weather context for each input feature,
     but it is downsampled by a factor of 4 in each spatial dimension,
      resulting in an input patch of 512×512 positions.
      In addition to the input patches having spatial dimensions,
       they also have a time dimension in the form of multiple time slices (see Supplement B.1 Table 4 for details.)
       This is to ensure that the network has accessto the temporal dynamics in the input features.
        After padding and concatenation together along the depth axis, the input sets are embedded
         using a convolutional recurrent network [32] in the time dimension
    """

    def __init__(self, in_channels, out_channels=384, ks=3, n_layers=1):
        super().__init__()
        self.rnn = ConvGRU(in_channels, out_channels, (ks, ks), n_layers, batch_first=True)

    def forward(self, x):
        x, h = self.rnn(x)
        return (x, h[-1])
