from metnet.layers.StochasticDepth import StochasticDepth
from metnet.layers.SqueezeExcitation import SqueezeExcite
from metnet.layers.MBConv import MBConv
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
