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
        # features: list[tuple[torch.Tensor, torch.Tensor]] = []
        cs_features: list[torch.Tensor] = []
        fs_features: list[torch.Tensor] = []

        # loop for each modalities
        for i, x in enumerate(x_in):
            cs, fs = self.msfes[i](x)
            cs_features.append(cs)
            fs_features.append(fs)
        
        # fuse features, features: list((b, f, 1400), (b, f, 499))
        # feature: tuple[torch.Tensor, torch.Tensor] = self.fusion(features)[0]
        cs_feature = torch.cat(cs_features, dim=1)
        fs_feature = torch.cat(fs_features, dim=1)

        # passing to sch
        y: torch.Tensor = self.sch((cs_feature, fs_feature))
        return y
