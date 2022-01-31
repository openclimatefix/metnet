import torch.nn as nn
import torch

class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'

    Paper: https://arxiv.org/pdf/1709.07871.pdf

    """
    def forward(self, x: torch.Tensor, gammas: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas
