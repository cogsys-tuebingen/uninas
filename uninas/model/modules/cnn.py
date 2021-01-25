import torch
import torch.nn as nn


class SqueezeModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()


class GapSqueezeModule(nn.Module):
    """ global average pooling and squeezing """

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gap(x).squeeze()


class PaddingToValueModule(nn.Module):
    def __init__(self, to_value: int, dim=-1):
        super().__init__()
        self.to_value = to_value
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        shape[self.dim] = self.to_value - shape[self.dim]
        return torch.cat([x, torch.zeros(shape, dtype=x.dtype, device=x.device)], dim=self.dim)
