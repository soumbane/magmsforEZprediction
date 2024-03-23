import torch
from torch.nn import functional as F


class PaddingLast1D(torch.nn.Module):
    size: int

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, self.size))
