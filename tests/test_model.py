import torch
import torch.nn.functional as F
from metnet import MetNet, MetNet2, MetNet3, MetNetPV


def test_metnet3_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MetNet3(
        # Use small sizes to keep test fast on CPU
        grid_height=16,
        grid_width=16,
        num_maxvit_blocks=1,
        num_resnet_blocks=1,
        crop_4km=16,
        crop_8km=8,
        crop_16km=4,
        crop_output=4,
    ).to(device)

    # Use small spatial size to avoid OOM on CPU
    high_res = torch.rand(1, 773, 64, 64).to(device)
    low_res = torch.rand(1, 17, 64, 64).to(device)

    out_surface, out_hrrr, out_precip = model(high_res, low_res, lead_time=0)

    # Check output shapes
    assert out_surface.shape[1] == 1460
    assert out_hrrr.shape[1] == 617
    assert out_precip.shape[1] == 1024

    # Check gradients flow through all parameters
    loss = out_surface.sum() + out_hrrr.sum() + out_precip.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

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
        12,
        16,
        16,
    )
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_metnet_backwards():
    model = MetNet(
        hidden_dim=32,
        forecast_steps=24,
        input_channels=16,
        output_channels=12,
        sat_channels=12,
        input_size=32,
    )
    # MetNet expects original HxW to be 4x the input size
    x = torch.randn((2, 12, 16, 128, 128))
    out = []
    for lead_time in range(24):
        out.append(model(x, lead_time))
    out = torch.stack(out, dim=1)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        24,
        12,
        8,
        8,
    )
    y = torch.randn((2, 24, 12, 8, 8))
    F.mse_loss(out, y).backward()
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_metnet_pv_backwards():
    model = MetNetPV(
        hidden_dim=128,
        num_att_layers=2,
        num_att_heads=16,
        forecast_steps=24,
        input_channels=16,
        output_channels=1,
        sat_channels=12,
        input_size=64,
        avg_pool_size=2,
        pv_fc_out_channels=128,
        pv_id_embedding_channels=256,
    )
    # MetNet expects original HxW to be 4x the input size
    x = torch.randn((2, 12, 16, 256, 256))
    pv_x = torch.randn((2, 12, 1, 1000))
    pv_idx = torch.randint(5000, (2, 1000))
    out = []
    for lead_time in range(24):
        out.append(model(x, pv_x, pv_idx, lead_time))
    out = torch.stack(out, dim=1)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (2, 24, 1)
    y = torch.randn((2, 24, 1))
    F.mse_loss(out, y).backward()
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
        12,
        128,
        128,
    )
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_metnet2_backward():
    model = MetNet2(
        forecast_steps=4,
        input_size=64,
        num_input_timesteps=6,
        upsampler_channels=128,
        lstm_channels=32,
        encoder_channels=64,
        center_crop_size=16,
    )
    # MetNet expects original HxW to be 4x the input size
    x = torch.randn((2, 6, 12, 256, 256))
    out = []
    for lead_time in range(4):
        out.append(model(x, lead_time))
    out = torch.stack(out, dim=1)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        4,
        12,
        64,
        64,
    )
    y = torch.rand((2, 4, 12, 64, 64))
    F.mse_loss(out, y).backward()
    assert not torch.isnan(out).any(), "Output included NaNs"
