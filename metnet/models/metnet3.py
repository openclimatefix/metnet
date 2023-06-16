"""MetNet-3"""
import torch
import torch.nn as nn
import torchvision.transforms
from huggingface_hub import PyTorchModelHubMixin

from metnet.layers import DownSampler, MetNetPreprocessor, TimeDistributed
from metnet.layers.ConditionWithTimeMetNet2 import ConditionWithTimeMetNet2
from metnet.layers.ConvLSTM import ConvLSTM
from metnet.layers.DilatedCondConv import DilatedResidualConv, UpsampleResidualConv


class MetNet3(torch.nn.Module, PyTorchModelHubMixin):
    """MetNet-3 model for weather forecasting"""
    def __init__(self, input_size: int = 624, sparse_input_channels: int = 14, dense_input_channels: int = 643,
                 context_input_channels: int = 17, low_res_output_size=(128, 128), low_res_output_channels=(14, 617),
                 high_res_output_size=(512,), high_res_output_channels=(1,), center_crop_size: int = 192, forecast_steps: int = 720):
        """
        MetNet-3, with defaults matching the paper as close as possible

        Args:
            input_size: Input size, in pixels, of the sparse and dense central inputs
            sparse_input_channels: Number of channels in the sparse input
            dense_input_channels: Number of channels in the dense input
            context_input_channels: Number of channels in the context input
            low_res_output_size: List of output sizes, in pixels, for the low resolution outputs, assuming square outputs
            low_res_output_channels: List of output channel number for each of those outputs
            high_res_output_size: List of output sizes, in pixels, for the high resolution outputs, assuming square outputs
            high_res_output_channels: List of number of channels for each of the high-resolution outputs
            center_crop_size: Center crop size of first center crop in pixel space. Second central crop will be half this
                size
            forecast_steps: Number of forecast steps to predict
        """
        super().__init__()

    def forward(self, sparse_input: torch.Tensor, dense_input: torch.Tensor, context_input: torch.Tensor, lead_time: int = 0) -> torch.Tensor:
        return NotImplementedError
