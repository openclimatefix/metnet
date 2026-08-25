from metnet.layers.InputEmbedding import InputEmbedding
from metnet.layers.MaxViT import MaxViTBlock, MaxViTDataClass, MetNetMaxVit
from metnet.layers.StochasticDepth import StochasticDepth
from metnet.layers.SqueezeExcitation import SqueezeExcite
from metnet.layers.MBConv import MBConv
from metnet.layers.MultiheadSelfAttention2D import MultiheadSelfAttention2D
from metnet.layers.PartitionAttention import BlockAttention, GridAttention
from metnet.layers.ConditionWithTimeMetNet3 import ConditionWithTimeMetNet3
from metnet.layers.LeadTimeConditioner import LeadTimeConditioner
from metnet.layers.TopographicalEmbedding import TopographicalEmbedding

import torch


def test_topographical_embedding_gradients():
    batch, channels, height, width = 2, 12, 624, 624
    x = torch.rand(batch, channels, height, width)

    embedding = TopographicalEmbedding(grid_height=624, grid_width=624)
    output = embedding(x)

    loss = output.sum()
    loss.backward()

    # Check output shapes
    assert output.shape == (batch, 20, height, width)

    # embedding_grid is a learned parameter so should have gradients
    for name, param in embedding.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_input_embedding_gradients():
    batch, in_channels, height, width = 2, 793, 16, 16
    x = torch.rand(batch, in_channels, height, width)

    embedding = InputEmbedding(in_channels=793)
    output = embedding(x)

    loss = output.sum()
    loss.backward()

    # Check output shapes
    assert output.shape == (batch, 512, height, width)

    for name, param in embedding.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_condition_with_time_metnet3():
    batch, channels, height, width = 2, 512, 16, 16
    test_tensor = torch.rand(batch, channels, height, width)

    conditioner = ConditionWithTimeMetNet3()

    # Check output shapes
    scale, bias = conditioner(test_tensor, timestep=0)
    assert scale.shape == (batch, 512)
    assert bias.shape == (batch, 512)

    # Check identity initialization — scale should be ~1, bias should be ~0
    assert torch.allclose(scale, torch.ones_like(scale))
    assert torch.allclose(bias, torch.zeros_like(bias))


def test_backward_pass_gradients_flow():
    """Ensure gradients flow through the lead time network on a backward pass."""
    batch, channels, height, width = 2, 512, 16, 16
    test_tensor = torch.rand(batch, channels, height, width)
    conditioner = ConditionWithTimeMetNet3()
    film = LeadTimeConditioner()

    scale, bias = conditioner(test_tensor, timestep=0)

    # Simulate FiLM conditioning, then reduce to a scalar loss
    conditioned = film(test_tensor, scale, bias)
    loss = conditioned.sum()
    loss.backward()

    for name, param in conditioner.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_stochastic_depth():
    test_tensor = torch.ones(1)

    stochastic_depth = StochasticDepth(drop_prob=0)
    assert test_tensor == stochastic_depth(test_tensor)

    stochastic_depth = StochasticDepth(drop_prob=1)
    assert torch.zeros_like(test_tensor) == stochastic_depth(test_tensor)


def test_squeeze_excitation():
    n, c, h, w = 1, 3, 16, 16
    test_tensor = torch.rand(n, c, h, w)

    squeeze_excite = SqueezeExcite(in_channels=c)
    assert test_tensor.shape == squeeze_excite(test_tensor).shape


def test_mbconv():
    n, c, h, w = 1, 3, 16, 16
    test_tensor = torch.rand(n, c, h, w)
    mb_conv = MBConv(c)

    assert test_tensor.shape == mb_conv(test_tensor).shape


def test_multiheaded_self_attention_2D():
    n, c, h, w = 1, 3, 16, 16
    test_tensor = torch.rand(n, c, h, w)
    rel_self_attention = MultiheadSelfAttention2D(c)
    assert test_tensor.shape == rel_self_attention(test_tensor).shape


def test_block_attention():
    n, c, h, w = 1, 3, 16, 16
    test_tensor = torch.rand(n, c, h, w)
    block_attention = BlockAttention(c)

    assert test_tensor.shape == block_attention(test_tensor).shape


def test_grid_attention():
    n, c, h, w = 1, 3, 16, 16
    test_tensor = torch.rand(n, c, h, w)
    grid_attention = GridAttention(c)

    assert test_tensor.shape == grid_attention(test_tensor).shape


def test_maxvitblock():
    n, c, h, w = 1, 3, 16, 16
    test_tensor = torch.rand(n, c, h, w)

    maxvit_block = MaxViTBlock(in_channels=c, maxvit_config=MaxViTDataClass())
    assert test_tensor.shape == maxvit_block(test_tensor).shape


def test_maxvitblock_applies_lead_time_conditioning():
    n, c, h, w = 2, 3, 16, 16
    test_tensor = torch.arange(n * c * h * w, dtype=torch.float32).reshape(n, c, h, w)
    scale = torch.tensor([[2.0] * c, [3.0] * c])
    bias = torch.tensor([[1.0] * c, [4.0] * c])

    maxvit_block = MaxViTBlock(in_channels=c, maxvit_config=MaxViTDataClass())
    maxvit_block.mb_conv = torch.nn.Identity()
    maxvit_block.block_attention.attention = torch.nn.Identity()
    maxvit_block.grid_attention.attention = torch.nn.Identity()

    normalized_inputs = []
    conditioned_inputs = []
    for attention in (maxvit_block.block_attention, maxvit_block.grid_attention):
        attention.pre_norm_layer.register_forward_hook(
            lambda _module, _args, output: normalized_inputs.append(output)
        )
        attention.attention.register_forward_pre_hook(
            lambda _module, args: conditioned_inputs.append(args[0])
        )

    maxvit_block(test_tensor, scale, bias)

    assert len(normalized_inputs) == len(conditioned_inputs) == 2
    for normalized, conditioned in zip(normalized_inputs, conditioned_inputs):
        partitions_per_sample = normalized.shape[0] // n
        partition_scale = scale.repeat_interleave(partitions_per_sample, dim=0)
        partition_bias = bias.repeat_interleave(partitions_per_sample, dim=0)
        expected = normalized * partition_scale[:, :, None, None]
        expected = expected + partition_bias[:, :, None, None]
        torch.testing.assert_close(conditioned, expected)


def test_metnet_maxvit():
    n, c, h, w = 1, 3, 16, 16
    test_tensor = torch.rand(n, c, h, w)

    metnet_maxvit = MetNetMaxVit(in_channels=c)
    assert test_tensor.shape == metnet_maxvit(test_tensor).shape
