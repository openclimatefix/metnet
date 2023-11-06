from metnet.layers.MaxViT import MaxViTBlock, MaxViTDataClass, MetNetMaxVit
from metnet.layers.StochasticDepth import StochasticDepth
from metnet.layers.SqueezeExcitation import SqueezeExcite
from metnet.layers.MBConv import MBConv
from metnet.layers.MultiheadSelfAttention2D import MultiheadSelfAttention2D
from metnet.layers.PartitionAttention import BlockAttention, GridAttention
import torch


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


def test_metnet_maxvit():
    n, c, h, w = 1, 3, 16, 16
    test_tensor = torch.rand(n, c, h, w)

    metnet_maxvit = MetNetMaxVit(in_channels=c)
    assert test_tensor.shape == metnet_maxvit(test_tensor).shape
