import torch

from metnet import MetNet, MetNet2


def test_metnet_creation():
    model = MetNet(
        hidden_dim=32,
        forecast_steps=24,
        input_channels=16,
        output_channels=12,
        sat_channels=12,
        input_size=64,
    )
    # MetNet expects original HxW to be 4x the input size
    x = torch.randn((2, 12, 16, 256, 256))
    model.eval()
    with torch.no_grad():
        out = model(x)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        24,
        12,
        16,
        16,
    )
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_load_metnet_hf():
    model = MetNet.from_pretrained("openclimatefix/metnet")


def test_load_metnet2_hf():
    model = MetNet2.from_pretrained("openclimatefix/metnet-2")


def test_metnet2_creation():
    model = MetNet2(
        forecast_steps=8,
        input_size=128,
        num_input_timesteps=6,
        upsampler_channels=128,
        lstm_channels=32,
        encoder_channels=64,
        center_crop_size=32,
    )
    # MetNet expects original HxW to be 4x the input size
    x = torch.randn((2, 6, 12, 512, 512))
    model.eval()
    with torch.no_grad():
        out = model(x)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        8,
        12,
        128,
        128,
    )
    assert not torch.isnan(out).any(), "Output included NaNs"
