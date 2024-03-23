from magnet.nn.fusion import Fusion

import torch


class MidConcat(Fusion):
    """
    Fusion at the middle for encoders with multiple outputs
    
    * extends: `Fusion`
    """
    def forward(self, x_in: list[tuple[torch.Tensor, ...]]) -> list[tuple[torch.Tensor, ...]]:
        return super().forward(x_in)
    
    def fuse(self, x_in: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
        # initialize
        assert len(x_in) > 0, "Fused inputs must have at least one target."
        x_to_fuse: list[list[torch.Tensor]] = [[] for _ in x_in[0]]

        # loop for each target
        for x in x_in:
            # loop each feature
            for i, f in enumerate(x):
                x_to_fuse[i].append(f)

        # mean fusion
        y: tuple[torch.Tensor, ...] = tuple([torch.cat(x, dim=1) for x in x_to_fuse])
        return y


class MidSingleConcat(Fusion):
    def fuse(self, x_in: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_in, dim=1)
