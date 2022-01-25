import torch

from metnet import MetNet


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
