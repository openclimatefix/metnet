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
        upsample_method: str = "interp",
        num_upsampler_blocks: int = 2,
        num_context_blocks: int = 3,
        num_input_timesteps: int = 13,
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
        if image_encoder in ["downsampler", "default"]:
            self.image_encoder = DownSampler(input_channels + forecast_steps)
        else:
            raise ValueError(f"Image_encoder {image_encoder} is not recognized")

        total_number_of_conv_blocks = num_context_blocks * len(encoder_dilations) + 8
        total_number_of_conv_blocks = total_number_of_conv_blocks + num_upsampler_blocks if upsample_method != 'interp' else total_number_of_conv_blocks
        total_number_of_conv_blocks += num_input_timesteps
        self.ct = ConditionWithTimeMetNet2(forecast_steps, total_blocks_to_condition = total_number_of_conv_blocks, hidden_dim = lead_time_features)

        # ConvLSTM with 13 timesteps, 128 LSTM channels, 18 encoder blocks, 384 encoder channels,
        self.conv_lstm = ConvLSTM(
            input_dim=input_channels, hidden_dim=lstm_channels, kernel_size=3, num_layers=num_input_timesteps
        )

        # Lead time network layers that generate a bias and scale vector for the lead time

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
        for block in range(num_context_blocks - 1):
            self.context_blocks.extend(
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
        if upsample_method == 'interp':
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            # The paper though, under the architecture, has 2 upsample blocks with 512 channels, indicating it might be a
            # transposed convolution instead?
            # 2 Upsample blocks with 512 channels
            self.upsample = nn.ModuleList(
                UpsampleResidualConv(
                    input_channels=encoder_channels, output_channels=upsampler_channels, kernel_size=3
                )
                for _ in range(num_upsampler_blocks)
            )
        self.upsample_method = upsample_method

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

    def forward(self, x: torch.Tensor):
        """It takes a rank 5 tensor
        - imgs [bs, seq_len, channels, h, w]
        """

        # Compute all timesteps, probably can be parallelized
        out = []
        x = self.preprocessor(x)
        for i in range(self.forecast_steps):
            # Compute scale and bias
            scale_and_bias = self.ct(x, i)
            block_num = 0

            #ConvLSTM

            # Context Stack
            for layer in self.context_block_one:
                scale, bias = scale_and_bias[:, block_num]
                res = layer(res, scale, bias)
                block_num += 1
            for layer in self.context_blocks:
                scale, bias = scale_and_bias[:, block_num]
                res = layer(res, scale, bias)
                block_num += 1

            # Get Center Crop
            res = self.center_crop(res)

            # Upsample
            if self.upsample_method == 'interp':
                res = self.upsample(res)
            else:
                for layer in self.upsample:
                    scale, bias = scale_and_bias[:, block_num]
                    res = layer(res, scale, bias)
                    block_num += 1

            # Shallow network
            for layer in self.residual_block_three:
                scale, bias = scale_and_bias[:, block_num]
                res = layer(res, scale, bias)
                block_num += 1

            # Return 1x1 Conv
            res = self.head(res)
            out.append(res)
        out = torch.stack(out, dim=1)

        # Softmax for rain forecasting
        return out


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


class ConditionWithTimeMetNet2(nn.Module):
    def __init__(self, forecast_steps, hidden_dim, total_blocks_to_condition, ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.total_blocks = total_blocks_to_condition
        self.lead_time_network = nn.ModuleList(
            [
                nn.Linear(in_features=forecast_steps, out_features=hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=total_blocks_to_condition*2),
                ]
            )

    def forward(self, x: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        Get the scale and bias for the conditioning layers

        Args:
            x: The Tensor that is used
            timestep: Index of the timestep to use, between 0 and forecast_steps

        Returns:
            Tensor of shape (Batch, blocks, 2)
        """
        # One hot encode the timestep
        timesteps = torch.zeros(x.size()[0], self.forecast_steps, dtype = torch.int16)
        timesteps[:, timestep] = 1
        # Get scales and biases
        scales_and_biases = self.lead_time_network(timesteps)
        scales_and_biases = einops.rearrange(scales_and_biases, 'b (block sb) -> b block sb', block=self.total_blocks, sb=2)
        return scales_and_biases
