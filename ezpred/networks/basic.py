import torch

from ezpred.nn import MSFE, SCH
from ezpred.nn.fusion import Fusion


class Basic(torch.nn.Module):
    msfes: torch.nn.ModuleList
    fusion: Fusion
    sch: SCH

    def __init__(self, *msfes: MSFE, fusion: Fusion, sch: SCH) -> None:
        super().__init__()
        self.msfes = torch.nn.ModuleList(msfes)
        self.fusion = fusion
        self.sch = sch

    def forward(self, x_in: list[torch.Tensor]) -> torch.Tensor:
        # initialize features
        features: list[tuple[torch.Tensor, torch.Tensor]] = []

        # loop for each modalities
        for i, x in enumerate(x_in):
            f = self.msfes[i](x)
            features.append(f)
        
        # fuse features, features: list((b, f, 1400), (b, f, 499))
        feature: tuple[torch.Tensor, torch.Tensor] = self.fusion(features)[0]

        # passing to sch
        y: torch.Tensor = self.sch(feature)
        return y
