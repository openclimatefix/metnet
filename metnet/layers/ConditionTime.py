import torch
from torch import nn as nn


def condition_time(x, i=0, size=(12, 16), seq_len=15):
    "create one hot encoded time image-layers, i in [1, seq_len]"
    assert i < seq_len
    times = (
        (torch.eye(seq_len, dtype=torch.long, device=x.device)[i])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    ones = torch.ones(1, *size, dtype=x.dtype, device=x.device)
    return times * ones


class ConditionTime(nn.Module):
    "Condition Time on a stack of images, adds `horizon` channels to image"

    def __init__(self, horizon, ch_dim=2, num_dims=5):
        super().__init__()
        self.horizon = horizon
        self.ch_dim = ch_dim
        self.num_dims = num_dims

    def forward(self, x, fstep=0):
        "x stack of images, fsteps"
        if self.num_dims == 5:
            bs, seq_len, ch, h, w = x.shape
            ct = condition_time(x, fstep, (h, w), seq_len=self.horizon).repeat(bs, seq_len, 1, 1, 1)
        else:
            bs, h, w, ch = x.shape
            ct = condition_time(x, fstep, (h, w), seq_len=self.horizon).repeat(bs, 1, 1, 1)
            ct = ct.permute(0, 2, 3, 1)
        x = torch.cat([x, ct], dim=self.ch_dim)
        assert x.shape[self.ch_dim] == (ch + self.horizon)  # check if it makes sense
        return x
