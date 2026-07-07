"""MetNet-3 model for weather forecasting."""

import torch
import torch.nn as nn
import torchvision
from huggingface_hub import PyTorchModelHubMixin

from metnet.layers.ConditionWithTimeMetNet3 import ConditionWithTimeMetNet3
from metnet.layers.DilatedCondConv import DilatedResidualConv
from metnet.layers.InputEmbedding import InputEmbedding
from metnet.layers.LeadTimeConditioner import LeadTimeConditioner
from metnet.layers.MaxViT import MetNetMaxVit
from metnet.layers.TopographicalEmbedding import TopographicalEmbedding


class MetNet3(torch.nn.Module, PyTorchModelHubMixin):
    """MetNet-3 model for weather forecasting."""

    def __init__(
        self,
        high_res_in_channels: int = 773,  # 4km path after concatenation
        low_res_in_channels: int = 17,  # 8km path (low-res radar + GOES)
        hidden_channels: int = 512,  # fixed throughout network per paper
        num_resnet_blocks: int = 2,  # per stage per diagram
        num_maxvit_blocks: int = 12,  # per diagram
        kernel_size: int = 3,  # fixed per paper
        grid_height: int = 624,
        grid_width: int = 624,
        crop_4km: int = 192,  # 768km ÷ 4km
        crop_8km: int = 96,  # 768km ÷ 8km
        crop_16km: int = 48,  # 768km ÷ 16km
        crop_output: int = 128,  # 512km ÷ 4km
        forecast_steps: int = 722,  # 0-24hrs at 2min intervals
        topographical_channels: int = 20,  # number of channels for topographical embedding
        surface_output_channels: int = 1460,  # 5 variables by 256 bins + 180 bins
        hrrr_output_channels: int = 617,  # deterministic HRRR state
        precip_output_channels: int = 1024,  # 2 channels by 512 bins
        **kwargs,
    ):
        """ "MetNet-3 model for weather forecasting up to 24 hours ahead.

        MetNet-3 is a model which learns from both dense and sparse data sensors

        Paper: https://arxiv.org/pdf/2306.06079

        The architecture of MetNet-3 differs from MetNet-2 in the following ways:
            - Replaces ConvLSTM with convolutional ResNet blocks
            - Replaces dilated convolutions with a MaxViT transformer at the bottleneck
            - Uses a single shared FiLM conditioning MLP (hidden size 32) instead of per-layer MLPs
            - Extends lead time from 12 to 24 hours (722 steps at 2 minute intervals)
            - Adds topographical embeddings learned end-to-end
            - Uses separate high-resolution (4km) and low-resolution (8km) input paths
            - 512 channels throughout the network

        Args:
            high_res_in_channels: Number of input channels for the 4km high-resolution path,
            low_res_in_channels: Number of input channels for the 8km low-resolution path,
            hidden_channels: Number of channels throughout the network, 512 as per the paper
            num_resnet_blocks: Number of ResNet blocks per stage, 2 as per the paper
            num_maxvit_blocks: Number of MaxViT blocks at the 16km bottleneck, 12 as per the paper
            kernel_size: Convolution kernel size, 3 as per the paper
            grid_height: Height of the topographical embedding grid in pixels
            grid_width: Width of the topographical embedding grid in pixels
            crop_4km: Center crop size at 4km resolution, 192 pixels (768km ÷ 4km)
            crop_8km: Center crop size at 8km resolution, 96 pixels (768km ÷ 8km)
            crop_16km: Center crop size at 16km resolution, 48 pixels (768km ÷ 16km)
            crop_output: Final output crop size, 128 pixels (512km ÷ 4km)
            topographical_channels: Number of channels for topological embedding
            forecast_steps: Number of lead time steps, 722 for 0-24 hours at 2 minute intervals
            surface_output_channels: Number of channels for surface variable predictions
            hrrr_output_channels: Number of channels for HRRR output predictions
            precip_output_channels:  Number of channels for precip output after 1km decoder
            **kwargs: dict[str, Unknown]
        """
        super().__init__()
        config = locals()
        config.pop("self")
        config.pop("__class__")
        self.config = kwargs.pop("config", config)

        # Extract from config (handles both direct instantiation and from_pretrained)
        high_res_in_channels = self.config["high_res_in_channels"]
        low_res_in_channels = self.config["low_res_in_channels"]
        hidden_channels = self.config["hidden_channels"]
        num_resnet_blocks = self.config["num_resnet_blocks"]
        num_maxvit_blocks = self.config["num_maxvit_blocks"]
        kernel_size = self.config["kernel_size"]
        grid_height = self.config["grid_height"]
        grid_width = self.config["grid_width"]
        crop_4km = self.config["crop_4km"]
        crop_8km = self.config["crop_8km"]
        crop_16km = self.config["crop_16km"]
        crop_output = self.config["crop_output"]
        topographical_channels = self.config["topographical_channels"]
        forecast_steps = self.config["forecast_steps"]
        surface_output_channels = self.config["surface_output_channels"]
        hrrr_output_channels = self.config["hrrr_output_channels"]
        precip_output_channels = self.config["precip_output_channels"]

        # Store what's needed in forward pass
        self.forecast_steps = forecast_steps
        self.hidden_channels = hidden_channels

        # Topological Embedding
        self.topographical_embedding = TopographicalEmbedding(
            grid_height=grid_height,
            grid_width=grid_width,
            embedding_dim=topographical_channels,
        )

        # 4km Embdedding (Sparse + Dense + Topological)
        self.high_res_embedding = InputEmbedding(
            in_channels=high_res_in_channels + topographical_channels,
            out_channels=hidden_channels,
        )

        # 8km Embedding
        self.low_res_embedding = InputEmbedding(
            in_channels=low_res_in_channels,
            out_channels=hidden_channels,
        )

        # Shared FiLM conditioner — single instance used throughout
        self.lead_time_conditioner = ConditionWithTimeMetNet3(
            forecast_steps=forecast_steps,
            num_feature_maps=hidden_channels,
        )

        # Applies the scale and bias from lead_time_conditioner to feature maps
        self.lead_time_applier = LeadTimeConditioner()

        # ResNet blocks at each stage
        self.resnet_blocks_4km = nn.ModuleList(
            [
                DilatedResidualConv(
                    input_channels=hidden_channels,
                    output_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=1,
                )
                for _ in range(num_resnet_blocks)
            ]
        )
        self.resnet_blocks_8km = nn.ModuleList(
            [
                DilatedResidualConv(
                    input_channels=hidden_channels * 2,  # first block gets concat features
                    output_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=1,
                ),
                *[
                    DilatedResidualConv(
                        input_channels=hidden_channels,
                        output_channels=hidden_channels,
                        kernel_size=kernel_size,
                        dilation=1,
                    )
                    for _ in range(num_resnet_blocks - 1)
                ],
            ]
        )

        self.resnet_blocks_8km_decoder = nn.ModuleList(
            [
                DilatedResidualConv(
                    input_channels=hidden_channels * 3,  # first block
                    output_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=1,
                ),
                *[
                    DilatedResidualConv(
                        input_channels=hidden_channels,
                        output_channels=hidden_channels,
                        kernel_size=kernel_size,
                        dilation=1,
                    )
                    for _ in range(num_resnet_blocks - 1)
                ],
            ]
        )

        self.resnet_blocks_4km_decoder = nn.ModuleList(
            [
                DilatedResidualConv(
                    input_channels=hidden_channels * 2,  # first block gets concat features
                    output_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=1,
                ),
                *[
                    DilatedResidualConv(
                        input_channels=hidden_channels,
                        output_channels=hidden_channels,
                        kernel_size=kernel_size,
                        dilation=1,
                    )
                    for _ in range(num_resnet_blocks - 1)
                ],
            ]
        )

        self.resnet_blocks_1km = nn.ModuleList(
            [
                DilatedResidualConv(
                    input_channels=hidden_channels,
                    output_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=1,
                )
                for _ in range(num_resnet_blocks)
            ]
        )

        # MaxViT bottleneck
        self.maxvit = MetNetMaxVit(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_blocks=num_maxvit_blocks,
        )

        # Upsampling
        self.upsample_16km = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_8km = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4km = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        # Downsampling — stride 2 conv halves spatial resolution
        self.downsample_4km = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=2,
            stride=2,
        )
        self.downsample_8km = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=2,
            stride=2,
        )
        # Croppping
        # where center_crop_size=192 (768km ÷ 4km)
        self.center_crop_4km = torchvision.transforms.CenterCrop(size=crop_4km)
        self.center_crop_8km = torchvision.transforms.CenterCrop(size=crop_8km)
        self.center_crop_16km = torchvision.transforms.CenterCrop(size=crop_16km)
        self.center_crop_output = torchvision.transforms.CenterCrop(size=crop_output)

        # Output heads
        # Surface + HRRR: MLP with hidden size 4096
        self.head_surface = nn.Sequential(
            nn.Conv2d(hidden_channels, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4096, surface_output_channels, kernel_size=1),
        )

        self.head_hrrr = nn.Sequential(
            nn.Conv2d(hidden_channels, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4096, hrrr_output_channels, kernel_size=1),
        )

        self.head_1km = nn.Sequential(
            nn.Conv2d(hidden_channels, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4096, precip_output_channels, kernel_size=1),
        )

    def forward(
        self, high_res_input: torch.Tensor, low_res_input: torch.Tensor, lead_time: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for MetNet-3.

        Args:
            high_res_input: High resolution input tensor [B, 773, 624, 624]
            low_res_input: Low resolution input tensor [B, 17, 624, 624]
            lead_time: Lead time index between 0 and forecast_steps

        Returns:
            Output tensor
        """
        ## Generate topographical embedding and concatenate with high res input
        topo = self.topographical_embedding(high_res_input)
        high_res_input = torch.cat([high_res_input, topo], dim=1)  # [B, 793, 624, 624]

        # Embed first to get to hidden_channels size
        x = self.high_res_embedding(high_res_input)  # [B, 512, 624, 624]
        low_res = self.low_res_embedding(low_res_input)  # [B, 512, 624, 624]

        # "The conditioning is applied to the network inputs and after each layer normalization."
        scale, bias = self.lead_time_conditioner(x, lead_time)

        # Apply conditioning to embedded inputs
        x = self.lead_time_applier(x, scale, bias)
        low_res = self.lead_time_applier(low_res, scale, bias)

        # --- 4km Encoder ---
        skip_4km = self.center_crop_4km(x)  # save for 4km decoder skip 624 → 192
        for block in self.resnet_blocks_4km:
            x = block(x, scale, bias)
        x = self.downsample_4km(x)  # 624 → 312

        # Pad dynamically to match low_res spatial size
        target_size = low_res.shape[-1]
        current_size = x.shape[-1]
        pad = (target_size - current_size) // 2
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        # --- 8km Encoder ---
        x = torch.cat([x, low_res], dim=1)  # concat along channels
        skip_8km = self.center_crop_8km(x)  # save for 8km decoder skip
        for block in self.resnet_blocks_8km:
            x = block(x, scale, bias)
        x = self.downsample_8km(x)  # 624 → 312

        # --- 16km Bottleneck ---
        x = self.maxvit(x)
        x = self.center_crop_16km(x)  # crop to 768² km → 48x48

        # --- 8km Decoder ---
        x = self.upsample_16km(x)  # 48 → 96
        x = torch.cat([x, skip_8km], dim=1)  # concat skip connection
        for block in self.resnet_blocks_8km_decoder:
            x = block(x, scale, bias)

        # --- 4km Decoder ---
        x = self.upsample_8km(x)  # 96 → 192
        x = torch.cat([x, skip_4km], dim=1)  # concat skip connection
        for block in self.resnet_blocks_4km_decoder:
            x = block(x, scale, bias)
        x = self.center_crop_output(x)  # crop to 512² km → 128x128

        # --- 4km Output heads ---
        out_surface = self.head_surface(x)  # surface weather variables
        out_hrrr = self.head_hrrr(x)  # assimilated weather state

        # --- 1km Decoder ---
        x = self.upsample_4km(x)  # 128 → 512
        for block in self.resnet_blocks_1km:
            x = block(x, scale, bias)

        # --- 1km Output head ---
        out_precip = self.head_1km(x)  # precipitation

        return out_surface, out_hrrr, out_precip
